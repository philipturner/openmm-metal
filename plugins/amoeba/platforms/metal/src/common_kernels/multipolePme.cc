#define ARRAY(x,y) array[(x)-1+((y)-1)*PME_ORDER]

/**
 * Calculate the spline coefficients for a single atom along a single axis.
 */
DEVICE void computeBSplinePoint(real4* thetai, real w, real* array) {
    // initialization to get to 2nd order recursion

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1 - w;

    // perform one pass to get to 3rd order recursion

    ARRAY(3,3) = 0.5f * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5f * ((1+w)*ARRAY(2,1)+(2-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5f * (1-w) * ARRAY(2,1);

    // compute standard B-spline recursion to desired order

    for (int i = 4; i <= PME_ORDER; i++)
    {
        int k = i - 1;
        real denom = RECIP((real) k);
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1-w) * ARRAY(k,1);
    }

    // get coefficients for the B-spline first derivative

    int k = PME_ORDER - 1;
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline second derivative

    k = PME_ORDER - 2;
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline third derivative

    k = PME_ORDER - 3;
    ARRAY(k,PME_ORDER-2) = ARRAY(k,PME_ORDER-3);
    for (int i = PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // copy coefficients from temporary to permanent storage

    for (int i = 1; i <= PME_ORDER; i++)
        thetai[i-1] = make_real4(ARRAY(PME_ORDER,i), ARRAY(PME_ORDER-1,i), ARRAY(PME_ORDER-2,i), ARRAY(PME_ORDER-3,i));
}

/**
 * Convert the fixed multipoles from Cartesian to fractional coordinates.
 */
KERNEL void transformMultipolesToFractionalCoordinates(GLOBAL const real* RESTRICT labDipole,
#ifdef HIPPO
        GLOBAL const real* RESTRICT labQXX, GLOBAL const real* RESTRICT labQXY, GLOBAL const real* RESTRICT labQXZ, GLOBAL const real* RESTRICT labQYY, GLOBAL const real* RESTRICT labQYZ,
#else
        GLOBAL const real* RESTRICT labQuadrupole,
#endif
        GLOBAL real* RESTRICT fracDipole, GLOBAL real* RESTRICT fracQuadrupole, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // Build matrices for transforming the dipoles and quadrupoles.
    
    LOCAL real a[3][3];
    if (LOCAL_ID == 0) {
        a[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        a[0][1] = GRID_SIZE_X*recipBoxVecY.x;
        a[0][2] = GRID_SIZE_X*recipBoxVecZ.x;
        a[1][0] = GRID_SIZE_Y*recipBoxVecX.y;
        a[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        a[1][2] = GRID_SIZE_Y*recipBoxVecZ.y;
        a[2][0] = GRID_SIZE_Z*recipBoxVecX.z;
        a[2][1] = GRID_SIZE_Z*recipBoxVecY.z;
        a[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    int index1[] = {0, 0, 0, 1, 1, 2};
    int index2[] = {0, 1, 2, 1, 2, 2};
    LOCAL real b[6][6];
    if (LOCAL_ID < 36) {
        int i = LOCAL_ID/6;
        int j = LOCAL_ID-6*i;
        b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
        if (index1[i] != index2[i])
            b[i][j] += a[index1[i]][index2[j]]*a[index2[i]][index1[j]];
    }
    SYNC_THREADS;
    
    // Transform the multipoles.
    
    real quadScale[] = {1, 2, 2, 1, 2, 1};
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        for (int j = 0; j < 3; j++) {
            real dipole = 0;
            for (int k = 0; k < 3; k++)
                dipole += a[j][k]*labDipole[3*i+k];
            fracDipole[3*i+j] = dipole;
        }
        for (int j = 0; j < 6; j++) {
#ifdef HIPPO
            real quadrupole = quadScale[0]*b[j][0]*labQXX[i] +
                              quadScale[1]*b[j][1]*labQXY[i] +
                              quadScale[2]*b[j][2]*labQXZ[i] +
                              quadScale[3]*b[j][3]*labQYY[i] +
                              quadScale[4]*b[j][4]*labQYZ[i] -
                              quadScale[5]*b[j][5]*(labQXX[i]+labQYY[i]);
#else
            real quadrupole = 0;
            for (int k = 0; k < 5; k++)
                quadrupole += quadScale[k]*b[j][k]*labQuadrupole[5*i+k];
            quadrupole -= quadScale[5]*b[j][5]*(labQuadrupole[5*i]+labQuadrupole[5*i+3]);
#endif
            fracQuadrupole[6*i+j] = quadrupole;
        }
    }
}

/**
 * Convert the potential from fractional to Cartesian coordinates.
 */
KERNEL void transformPotentialToCartesianCoordinates(GLOBAL const real* RESTRICT fphi, GLOBAL real* RESTRICT cphi, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // Build matrices for transforming the potential.

    LOCAL real a[3][3];
    if (LOCAL_ID == 0) {
        a[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        a[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        a[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        a[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        a[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        a[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        a[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        a[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        a[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    int index1[] = {0, 1, 2, 0, 0, 1};
    int index2[] = {0, 1, 2, 1, 2, 2};
    LOCAL real b[6][6];
    if (LOCAL_ID < 36) {
        int i = LOCAL_ID/6;
        int j = LOCAL_ID-6*i;
        b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
        if (index1[j] != index2[j])
            b[i][j] += (i < 3 ? b[i][j] : a[index1[i]][index2[j]]*a[index2[i]][index1[j]]);
    }
    SYNC_THREADS;

    // Transform the potential.
    
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        cphi[10*i] = fphi[i];
        cphi[10*i+1] = a[0][0]*fphi[i+NUM_ATOMS*1] + a[0][1]*fphi[i+NUM_ATOMS*2] + a[0][2]*fphi[i+NUM_ATOMS*3];
        cphi[10*i+2] = a[1][0]*fphi[i+NUM_ATOMS*1] + a[1][1]*fphi[i+NUM_ATOMS*2] + a[1][2]*fphi[i+NUM_ATOMS*3];
        cphi[10*i+3] = a[2][0]*fphi[i+NUM_ATOMS*1] + a[2][1]*fphi[i+NUM_ATOMS*2] + a[2][2]*fphi[i+NUM_ATOMS*3];
        for (int j = 0; j < 6; j++) {
            cphi[10*i+4+j] = 0;
            for (int k = 0; k < 6; k++)
                cphi[10*i+4+j] += b[j][k]*fphi[i+NUM_ATOMS*(4+k)];
        }
    }
}

KERNEL void gridSpreadFixedMultipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT fracDipole,
        GLOBAL const real* RESTRICT fracQuadrupole,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#elif defined(HIPPO)
        GLOBAL real* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
#ifdef HIPPO
        GLOBAL const real* RESTRICT coreCharge, GLOBAL const real* RESTRICT valenceCharge,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    // We have shared memory to spare, and putting the workspace array there reduces the load on L2 cache.
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);
#ifdef HIPPO
        real atomCharge = coreCharge[m]+valenceCharge[m];
#else
        real atomCharge = pos.w;
#endif
        real atomDipoleX = fracDipole[m*3];
        real atomDipoleY = fracDipole[m*3+1];
        real atomDipoleZ = fracDipole[m*3+2];
        real atomQuadrupoleXX = fracQuadrupole[m*6];
        real atomQuadrupoleXY = fracQuadrupole[m*6+1];
        real atomQuadrupoleXZ = fracQuadrupole[m*6+2];
        real atomQuadrupoleYY = fracQuadrupole[m*6+3];
        real atomQuadrupoleYZ = fracQuadrupole[m*6+4];
        real atomQuadrupoleZZ = fracQuadrupole[m*6+5];

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);
        
        // Spread the charge from this atom onto each grid point.
         
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = igrid1+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real4 t = theta1[ix];
            
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = igrid2+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real4 u = theta2[iy];
                real term0 = atomCharge*t.x*u.x + atomDipoleY*t.x*u.y + atomQuadrupoleYY*t.x*u.z + atomDipoleX*t.y*u.x + atomQuadrupoleXY*t.y*u.y + atomQuadrupoleXX*t.z*u.x;
                real term1 = atomDipoleZ*t.x*u.x + atomQuadrupoleYZ*t.x*u.y + atomQuadrupoleXZ*t.y*u.x;
                real term2 = atomQuadrupoleZZ*t.x*u.x;
                
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real4 v = theta3[iz];
                    real add = term0*v.x + term1*v.y + term2*v.z;
#ifdef HIPPO
    #ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add));
    #else
                    ATOMIC_ADD(&pmeGrid[index], add);
    #endif
#else
    #ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add));
    #else
                    ATOMIC_ADD(&pmeGrid[index].x, add);
    #endif
#endif
                }
            }
        }
    }
}

KERNEL void gridSpreadInducedDipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
#ifndef HIPPO
        GLOBAL const real* RESTRICT inducedDipolePolar,
#endif
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#elif defined(HIPPO)
        GLOBAL real* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    // We have shared memory to spare, and putting the workspace array there reduces the load on L2 cache.
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    LOCAL real cartToFrac[3][3];
    if (LOCAL_ID == 0) {
        cartToFrac[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        cartToFrac[0][1] = GRID_SIZE_X*recipBoxVecY.x;
        cartToFrac[0][2] = GRID_SIZE_X*recipBoxVecZ.x;
        cartToFrac[1][0] = GRID_SIZE_Y*recipBoxVecX.y;
        cartToFrac[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        cartToFrac[1][2] = GRID_SIZE_Y*recipBoxVecZ.y;
        cartToFrac[2][0] = GRID_SIZE_Z*recipBoxVecX.z;
        cartToFrac[2][1] = GRID_SIZE_Z*recipBoxVecY.z;
        cartToFrac[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);
        real3 cinducedDipole = make_real3(inducedDipole[3*m], inducedDipole[3*m+1], inducedDipole[3*m+2]);
        real3 finducedDipole = make_real3(cinducedDipole.x*cartToFrac[0][0] + cinducedDipole.y*cartToFrac[0][1] + cinducedDipole.z*cartToFrac[0][2],
                                          cinducedDipole.x*cartToFrac[1][0] + cinducedDipole.y*cartToFrac[1][1] + cinducedDipole.z*cartToFrac[1][2],
                                          cinducedDipole.x*cartToFrac[2][0] + cinducedDipole.y*cartToFrac[2][1] + cinducedDipole.z*cartToFrac[2][2]);
#ifndef HIPPO
        real3 cinducedDipolePolar = make_real3(inducedDipolePolar[3*m], inducedDipolePolar[3*m+1], inducedDipolePolar[3*m+2]);
        real3 finducedDipolePolar = make_real3(cinducedDipolePolar.x*cartToFrac[0][0] + cinducedDipolePolar.y*cartToFrac[0][1] + cinducedDipolePolar.z*cartToFrac[0][2],
                                               cinducedDipolePolar.x*cartToFrac[1][0] + cinducedDipolePolar.y*cartToFrac[1][1] + cinducedDipolePolar.z*cartToFrac[1][2],
                                               cinducedDipolePolar.x*cartToFrac[2][0] + cinducedDipolePolar.y*cartToFrac[2][1] + cinducedDipolePolar.z*cartToFrac[2][2]);
#endif

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);
        
        // Spread the charge from this atom onto each grid point.
         
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = igrid1+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real4 t = theta1[ix];
            
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = igrid2+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real4 u = theta2[iy];
                real term01 = finducedDipole.y*t.x*u.y + finducedDipole.x*t.y*u.x;
                real term11 = finducedDipole.z*t.x*u.x;
#ifndef HIPPO
                real term02 = finducedDipolePolar.y*t.x*u.y + finducedDipolePolar.x*t.y*u.x;
                real term12 = finducedDipolePolar.z*t.x*u.x;
#endif                
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real4 v = theta3[iz];

                    real add1 = term01*v.x + term11*v.y;
#ifdef HIPPO
    #ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add1));
    #else
                    ATOMIC_ADD(&pmeGrid[index], add1);
    #endif
#else
                    real add2 = term02*v.x + term12*v.y;
    #ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add1));
                    ATOMIC_ADD(&pmeGrid[2*index+1], (mm_ulong) realToFixedPoint(add2));
    #else
                    ATOMIC_ADD(&pmeGrid[index].x, add1);
                    ATOMIC_ADD(&pmeGrid[index].y, add2);
    #endif
#endif
                }
            }
        }
    }
}

/**
 * In double precision, we have to use fixed point to accumulate the grid values, so convert them to floating point.
 */
KERNEL void finishSpreadCharge(GLOBAL const mm_long* RESTRICT pmeGridLong, GLOBAL real* RESTRICT pmeGrid) {
#ifdef HIPPO
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
#else
    const unsigned int gridSize = 2*GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
#endif
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE)
        pmeGrid[index] = scale*pmeGridLong[index];
}

KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ, real4 periodicBoxSize,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#ifdef HIPPO
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
#else
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
#endif
    real expFactor = M_PI*M_PI/(EWALD_ALPHA*EWALD_ALPHA);
    real scaleFactor = RECIP(M_PI*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
#ifdef HIPPO
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
        int ky = remainder/(GRID_SIZE_Z/2+1);
        int kz = remainder-ky*(GRID_SIZE_Z/2+1);
#else
        int kx = index/(GRID_SIZE_Y*GRID_SIZE_Z);
        int remainder = index-kx*GRID_SIZE_Y*GRID_SIZE_Z;
        int ky = remainder/GRID_SIZE_Z;
        int kz = remainder-ky*GRID_SIZE_Z;
#endif
        if (kx == 0 && ky == 0 && kz == 0) {
            pmeGrid[index] = make_real2(0, 0);
            continue;
        }
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real denom = m2*bx*by*bz;
        real eterm = scaleFactor*EXP(-expFactor*m2)/denom;
        pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
    }
}

KERNEL void computeFixedPotentialFromGrid(
#ifdef HIPPO
        GLOBAL const real* RESTRICT pmeGrid,
#else
        GLOBAL const real2* RESTRICT pmeGrid,
#endif
        GLOBAL real* RESTRICT phi, GLOBAL mm_long* RESTRICT fieldBuffers,
#ifndef HIPPO
        GLOBAL mm_long* RESTRICT fieldPolarBuffers,
#endif
        GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labDipole, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    // We have shared memory to spare, and putting the workspace array there reduces the load on L2 cache.
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Compute the potential from this grid point.

        real tuv000 = 0;
        real tuv001 = 0;
        real tuv010 = 0;
        real tuv100 = 0;
        real tuv200 = 0;
        real tuv020 = 0;
        real tuv002 = 0;
        real tuv110 = 0;
        real tuv101 = 0;
        real tuv011 = 0;
        real tuv300 = 0;
        real tuv030 = 0;
        real tuv003 = 0;
        real tuv210 = 0;
        real tuv201 = 0;
        real tuv120 = 0;
        real tuv021 = 0;
        real tuv102 = 0;
        real tuv012 = 0;
        real tuv111 = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            real tu20 = 0;
            real tu11 = 0;
            real tu02 = 0;
            real tu30 = 0;
            real tu21 = 0;
            real tu12 = 0;
            real tu03 = 0;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
                real4 t = make_real4(0, 0, 0, 0);
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
#ifdef HIPPO
                    real tq = pmeGrid[gridIndex];
#else
                    real tq = pmeGrid[gridIndex].x;
#endif
                    real4 tadd = theta3[iz];
                    t.x += tq*tadd.x;
                    t.y += tq*tadd.y;
                    t.z += tq*tadd.z;
                    t.w += tq*tadd.w;
                }
                tu00 += u.x*t.x;
                tu10 += u.y*t.x;
                tu01 += u.x*t.y;
                tu20 += u.z*t.x;
                tu11 += u.y*t.y;
                tu02 += u.x*t.z;
                tu30 += u.w*t.x;
                tu21 += u.z*t.y;
                tu12 += u.y*t.z;
                tu03 += u.x*t.w;
            }
            tuv000 += v.x*tu00;
            tuv100 += v.y*tu00;
            tuv010 += v.x*tu10;
            tuv001 += v.x*tu01;
            tuv200 += v.z*tu00;
            tuv020 += v.x*tu20;
            tuv002 += v.x*tu02;
            tuv110 += v.y*tu10;
            tuv101 += v.y*tu01;
            tuv011 += v.x*tu11;
            tuv300 += v.w*tu00;
            tuv030 += v.x*tu30;
            tuv003 += v.x*tu03;
            tuv210 += v.z*tu10;
            tuv201 += v.z*tu01;
            tuv120 += v.y*tu20;
            tuv021 += v.x*tu21;
            tuv102 += v.y*tu02;
            tuv012 += v.x*tu12;
            tuv111 += v.y*tu11;
        }
        phi[m] = tuv000;
        phi[m+NUM_ATOMS] = tuv100;
        phi[m+NUM_ATOMS*2] = tuv010;
        phi[m+NUM_ATOMS*3] = tuv001;
        phi[m+NUM_ATOMS*4] = tuv200;
        phi[m+NUM_ATOMS*5] = tuv020;
        phi[m+NUM_ATOMS*6] = tuv002;
        phi[m+NUM_ATOMS*7] = tuv110;
        phi[m+NUM_ATOMS*8] = tuv101;
        phi[m+NUM_ATOMS*9] = tuv011;
        phi[m+NUM_ATOMS*10] = tuv300;
        phi[m+NUM_ATOMS*11] = tuv030;
        phi[m+NUM_ATOMS*12] = tuv003;
        phi[m+NUM_ATOMS*13] = tuv210;
        phi[m+NUM_ATOMS*14] = tuv201;
        phi[m+NUM_ATOMS*15] = tuv120;
        phi[m+NUM_ATOMS*16] = tuv021;
        phi[m+NUM_ATOMS*17] = tuv102;
        phi[m+NUM_ATOMS*18] = tuv012;
        phi[m+NUM_ATOMS*19] = tuv111;
        real dipoleScale = (4/(real) 3)*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/SQRT_PI;
        mm_long fieldx = realToFixedPoint(dipoleScale*labDipole[m*3]-tuv100*fracToCart[0][0]-tuv010*fracToCart[0][1]-tuv001*fracToCart[0][2]);
        mm_long fieldy = realToFixedPoint(dipoleScale*labDipole[m*3+1]-tuv100*fracToCart[1][0]-tuv010*fracToCart[1][1]-tuv001*fracToCart[1][2]);
        mm_long fieldz = realToFixedPoint(dipoleScale*labDipole[m*3+2]-tuv100*fracToCart[2][0]-tuv010*fracToCart[2][1]-tuv001*fracToCart[2][2]);
        fieldBuffers[m] = fieldx;
        fieldBuffers[m+PADDED_NUM_ATOMS] = fieldy;
        fieldBuffers[m+2*PADDED_NUM_ATOMS] = fieldz;
#ifndef HIPPO
        fieldPolarBuffers[m] = fieldx;
        fieldPolarBuffers[m+PADDED_NUM_ATOMS] = fieldy;
        fieldPolarBuffers[m+2*PADDED_NUM_ATOMS] = fieldz;
#endif
    }
}

KERNEL void computeInducedPotentialFromGrid(
#ifdef HIPPO
        GLOBAL const real* RESTRICT pmeGrid, GLOBAL real* RESTRICT extrapolatedPhi, int order,
#else
        GLOBAL const real2* RESTRICT pmeGrid, GLOBAL real* RESTRICT phid, GLOBAL real* RESTRICT phip,
#endif
        GLOBAL real* RESTRICT phidp, GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX,
        real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    // We have shared memory to spare, and putting the workspace array there reduces the load on L2 cache.
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Compute the potential from this grid point.

#ifndef HIPPO
        real tuv100_1 = 0;
        real tuv010_1 = 0;
        real tuv001_1 = 0;
        real tuv200_1 = 0;
        real tuv020_1 = 0;
        real tuv002_1 = 0;
        real tuv110_1 = 0;
        real tuv101_1 = 0;
        real tuv011_1 = 0;
        real tuv100_2 = 0;
        real tuv010_2 = 0;
        real tuv001_2 = 0;
        real tuv200_2 = 0;
        real tuv020_2 = 0;
        real tuv002_2 = 0;
        real tuv110_2 = 0;
        real tuv101_2 = 0;
        real tuv011_2 = 0;
#endif
        real tuv000 = 0;
        real tuv001 = 0;
        real tuv010 = 0;
        real tuv100 = 0;
        real tuv200 = 0;
        real tuv020 = 0;
        real tuv002 = 0;
        real tuv110 = 0;
        real tuv101 = 0;
        real tuv011 = 0;
        real tuv300 = 0;
        real tuv030 = 0;
        real tuv003 = 0;
        real tuv210 = 0;
        real tuv201 = 0;
        real tuv120 = 0;
        real tuv021 = 0;
        real tuv102 = 0;
        real tuv012 = 0;
        real tuv111 = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];
#ifndef HIPPO
            real tu00_1 = 0;
            real tu01_1 = 0;
            real tu10_1 = 0;
            real tu20_1 = 0;
            real tu11_1 = 0;
            real tu02_1 = 0;
            real tu00_2 = 0;
            real tu01_2 = 0;
            real tu10_2 = 0;
            real tu20_2 = 0;
            real tu11_2 = 0;
            real tu02_2 = 0;
#endif
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            real tu20 = 0;
            real tu11 = 0;
            real tu02 = 0;
            real tu30 = 0;
            real tu21 = 0;
            real tu12 = 0;
            real tu03 = 0;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
#ifdef HIPPO
                real t0 = 0;
                real t1 = 0;
                real t2 = 0;
#else
                real t0_1 = 0;
                real t1_1 = 0;
                real t2_1 = 0;
                real t0_2 = 0;
                real t1_2 = 0;
                real t2_2 = 0;
#endif
                real t3 = 0;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
                    real4 tadd = theta3[iz];
#ifdef HIPPO
                    real tq = pmeGrid[gridIndex];
                    t0 += tq*tadd.x;
                    t1 += tq*tadd.y;
                    t2 += tq*tadd.z;
                    t3 += tq*tadd.w;
#else
                    real2 tq = pmeGrid[gridIndex];
                    t0_1 += tq.x*tadd.x;
                    t1_1 += tq.x*tadd.y;
                    t2_1 += tq.x*tadd.z;
                    t0_2 += tq.y*tadd.x;
                    t1_2 += tq.y*tadd.y;
                    t2_2 += tq.y*tadd.z;
                    t3 += (tq.x+tq.y)*tadd.w;
#endif
                }
#ifndef HIPPO
                tu00_1 += u.x*t0_1;
                tu10_1 += u.y*t0_1;
                tu01_1 += u.x*t1_1;
                tu20_1 += u.z*t0_1;
                tu11_1 += u.y*t1_1;
                tu02_1 += u.x*t2_1;
                tu00_2 += u.x*t0_2;
                tu10_2 += u.y*t0_2;
                tu01_2 += u.x*t1_2;
                tu20_2 += u.z*t0_2;
                tu11_2 += u.y*t1_2;
                tu02_2 += u.x*t2_2;
                real t0 = t0_1 + t0_2;
                real t1 = t1_1 + t1_2;
                real t2 = t2_1 + t2_2;
#endif
                tu00 += u.x*t0;
                tu10 += u.y*t0;
                tu01 += u.x*t1;
                tu20 += u.z*t0;
                tu11 += u.y*t1;
                tu02 += u.x*t2;
                tu30 += u.w*t0;
                tu21 += u.z*t1;
                tu12 += u.y*t2;
                tu03 += u.x*t3;
            }
#ifndef HIPPO
            tuv100_1 += v.y*tu00_1;
            tuv010_1 += v.x*tu10_1;
            tuv001_1 += v.x*tu01_1;
            tuv200_1 += v.z*tu00_1;
            tuv020_1 += v.x*tu20_1;
            tuv002_1 += v.x*tu02_1;
            tuv110_1 += v.y*tu10_1;
            tuv101_1 += v.y*tu01_1;
            tuv011_1 += v.x*tu11_1;
            tuv100_2 += v.y*tu00_2;
            tuv010_2 += v.x*tu10_2;
            tuv001_2 += v.x*tu01_2;
            tuv200_2 += v.z*tu00_2;
            tuv020_2 += v.x*tu20_2;
            tuv002_2 += v.x*tu02_2;
            tuv110_2 += v.y*tu10_2;
            tuv101_2 += v.y*tu01_2;
            tuv011_2 += v.x*tu11_2;
#endif
            tuv000 += v.x*tu00;
            tuv100 += v.y*tu00;
            tuv010 += v.x*tu10;
            tuv001 += v.x*tu01;
            tuv200 += v.z*tu00;
            tuv020 += v.x*tu20;
            tuv002 += v.x*tu02;
            tuv110 += v.y*tu10;
            tuv101 += v.y*tu01;
            tuv011 += v.x*tu11;
            tuv300 += v.w*tu00;
            tuv030 += v.x*tu30;
            tuv003 += v.x*tu03;
            tuv210 += v.z*tu10;
            tuv201 += v.z*tu01;
            tuv120 += v.y*tu20;
            tuv021 += v.x*tu21;
            tuv102 += v.y*tu02;
            tuv012 += v.x*tu12;
            tuv111 += v.y*tu11;
        }
#ifndef HIPPO
        phid[m]   = 0;
        phid[m+NUM_ATOMS] = tuv100_1;
        phid[m+NUM_ATOMS*2] = tuv010_1;
        phid[m+NUM_ATOMS*3] = tuv001_1;
        phid[m+NUM_ATOMS*4] = tuv200_1;
        phid[m+NUM_ATOMS*5] = tuv020_1;
        phid[m+NUM_ATOMS*6] = tuv002_1;
        phid[m+NUM_ATOMS*7] = tuv110_1;
        phid[m+NUM_ATOMS*8] = tuv101_1;
        phid[m+NUM_ATOMS*9] = tuv011_1;

        phip[m]   = 0;
        phip[m+NUM_ATOMS] = tuv100_2;
        phip[m+NUM_ATOMS*2] = tuv010_2;
        phip[m+NUM_ATOMS*3] = tuv001_2;
        phip[m+NUM_ATOMS*4] = tuv200_2;
        phip[m+NUM_ATOMS*5] = tuv020_2;
        phip[m+NUM_ATOMS*6] = tuv002_2;
        phip[m+NUM_ATOMS*7] = tuv110_2;
        phip[m+NUM_ATOMS*8] = tuv101_2;
        phip[m+NUM_ATOMS*9] = tuv011_2;
#endif
        phidp[m] = tuv000;
        phidp[m+NUM_ATOMS*1] = tuv100;
        phidp[m+NUM_ATOMS*2] = tuv010;
        phidp[m+NUM_ATOMS*3] = tuv001;
        phidp[m+NUM_ATOMS*4] = tuv200;
        phidp[m+NUM_ATOMS*5] = tuv020;
        phidp[m+NUM_ATOMS*6] = tuv002;
        phidp[m+NUM_ATOMS*7] = tuv110;
        phidp[m+NUM_ATOMS*8] = tuv101;
        phidp[m+NUM_ATOMS*9] = tuv011;
        phidp[m+NUM_ATOMS*10] = tuv300;
        phidp[m+NUM_ATOMS*11] = tuv030;
        phidp[m+NUM_ATOMS*12] = tuv003;
        phidp[m+NUM_ATOMS*13] = tuv210;
        phidp[m+NUM_ATOMS*14] = tuv201;
        phidp[m+NUM_ATOMS*15] = tuv120;
        phidp[m+NUM_ATOMS*16] = tuv021;
        phidp[m+NUM_ATOMS*17] = tuv102;
        phidp[m+NUM_ATOMS*18] = tuv012;
        phidp[m+NUM_ATOMS*19] = tuv111;
#ifdef HIPPO
        extrapolatedPhi[10*NUM_ATOMS*order+m] = tuv000;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*1] = tuv100;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*2] = tuv010;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*3] = tuv001;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*4] = tuv200;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*5] = tuv020;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*6] = tuv002;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*7] = tuv110;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*8] = tuv101;
        extrapolatedPhi[10*NUM_ATOMS*order+m+NUM_ATOMS*9] = tuv011;
#endif
    }
}

KERNEL void computeFixedMultipoleForceAndEnergy(GLOBAL real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mm_long* RESTRICT torqueBuffers, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT labDipole,
#ifdef HIPPO
        GLOBAL const real* RESTRICT coreCharge, GLOBAL const real* RESTRICT valenceCharge, GLOBAL const real* RESTRICT labQXX,
        GLOBAL const real* RESTRICT labQXY, GLOBAL const real* RESTRICT labQXZ, GLOBAL const real* RESTRICT labQYY, GLOBAL const real* RESTRICT labQYZ,
#else
        GLOBAL const real* RESTRICT labQuadrupole,
#endif
        GLOBAL const real* RESTRICT fracDipole, GLOBAL const real* RESTRICT fracQuadrupole,
        GLOBAL const real* RESTRICT phi, GLOBAL const real* RESTRICT cphi_global, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real multipole[10];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
    mixed energy = 0;
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        // Compute the torque.

        multipole[1] = labDipole[i*3];
        multipole[2] = labDipole[i*3+1];
        multipole[3] = labDipole[i*3+2];
#ifdef HIPPO
        multipole[0] = coreCharge[i]+valenceCharge[i];
        multipole[4] = labQXX[i];
        multipole[5] = labQYY[i];
        multipole[7] = 2*labQXY[i];
        multipole[8] = 2*labQXZ[i];
        multipole[9] = 2*labQYZ[i];
#else
        multipole[0] = posq[i].w;
        multipole[4] = labQuadrupole[i*5];
        multipole[5] = labQuadrupole[i*5+3];
        multipole[7] = 2*labQuadrupole[i*5+1];
        multipole[8] = 2*labQuadrupole[i*5+2];
        multipole[9] = 2*labQuadrupole[i*5+4];
#endif
        multipole[6] = -(multipole[4]+multipole[5]);

        GLOBAL const real* cphi = &cphi_global[10*i];

        torqueBuffers[i] = realToFixedPoint(EPSILON_FACTOR*(multipole[3]*cphi[2] - multipole[2]*cphi[3]
                      + 2*(multipole[6]-multipole[5])*cphi[9]
                      + multipole[8]*cphi[7] + multipole[9]*cphi[5]
                      - multipole[7]*cphi[8] - multipole[9]*cphi[6]));

        torqueBuffers[i+PADDED_NUM_ATOMS] = realToFixedPoint(EPSILON_FACTOR*(multipole[1]*cphi[3] - multipole[3]*cphi[1]
                      + 2*(multipole[4]-multipole[6])*cphi[8]
                      + multipole[7]*cphi[9] + multipole[8]*cphi[6]
                      - multipole[8]*cphi[4] - multipole[9]*cphi[7]));

        torqueBuffers[i+PADDED_NUM_ATOMS*2] = realToFixedPoint(EPSILON_FACTOR*(multipole[2]*cphi[1] - multipole[1]*cphi[2]
                      + 2*(multipole[5]-multipole[4])*cphi[7]
                      + multipole[7]*cphi[4] + multipole[9]*cphi[8]
                      - multipole[7]*cphi[5] - multipole[8]*cphi[9]));

        // Compute the force and energy.

        multipole[1] = fracDipole[i*3];
        multipole[2] = fracDipole[i*3+1];
        multipole[3] = fracDipole[i*3+2];
        multipole[4] = fracQuadrupole[i*6];
        multipole[5] = fracQuadrupole[i*6+3];
        multipole[6] = fracQuadrupole[i*6+5];
        multipole[7] = fracQuadrupole[i*6+1];
        multipole[8] = fracQuadrupole[i*6+2];
        multipole[9] = fracQuadrupole[i*6+4];

        real3 f = make_real3(0);
        for (int k = 0; k < 10; k++) {
            energy += multipole[k]*phi[i+NUM_ATOMS*k];
            f.x += multipole[k]*phi[i+NUM_ATOMS*deriv1[k]];
            f.y += multipole[k]*phi[i+NUM_ATOMS*deriv2[k]];
            f.z += multipole[k]*phi[i+NUM_ATOMS*deriv3[k]];
        }
        f = make_real3(EPSILON_FACTOR*(f.x*fracToCart[0][0] + f.y*fracToCart[0][1] + f.z*fracToCart[0][2]),
                       EPSILON_FACTOR*(f.x*fracToCart[1][0] + f.y*fracToCart[1][1] + f.z*fracToCart[1][2]),
                       EPSILON_FACTOR*(f.x*fracToCart[2][0] + f.y*fracToCart[2][1] + f.z*fracToCart[2][2]));
        forceBuffers[i] -= (mm_ulong) realToFixedPoint(f.x);
        forceBuffers[i+PADDED_NUM_ATOMS] -= (mm_ulong) realToFixedPoint(f.y);
        forceBuffers[i+PADDED_NUM_ATOMS*2] -= (mm_ulong) realToFixedPoint(f.z);
    }
    energyBuffer[GLOBAL_ID] += 0.5f*EPSILON_FACTOR*energy;
}

KERNEL void computeInducedDipoleForceAndEnergy(GLOBAL real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mm_long* RESTRICT torqueBuffers, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT labDipole,
#ifdef HIPPO
        GLOBAL const real* RESTRICT coreCharge, GLOBAL const real* RESTRICT valenceCharge, GLOBAL const real* RESTRICT extrapolatedDipole,
        GLOBAL const real* RESTRICT extrapolatedPhi, GLOBAL const real* RESTRICT labQXX, GLOBAL const real* RESTRICT labQXY,
        GLOBAL const real* RESTRICT labQXZ, GLOBAL const real* RESTRICT labQYY, GLOBAL const real* RESTRICT labQYZ,
#else
        GLOBAL const real* RESTRICT labQuadrupole,
#endif
        GLOBAL const real* RESTRICT fracDipole, GLOBAL const real* RESTRICT fracQuadrupole, GLOBAL const real* RESTRICT inducedDipole_global,
#ifndef HIPPO
        GLOBAL const real* RESTRICT inducedDipolePolar_global,
#endif
        GLOBAL const real* RESTRICT phi,
#ifndef HIPPO
        GLOBAL const real* RESTRICT phid, GLOBAL const real* RESTRICT phip,
#endif
        GLOBAL const real* RESTRICT phidp, GLOBAL const real* RESTRICT cphi_global, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real multipole[10];
    real cinducedDipole[3], inducedDipole[3];
    real cinducedDipolePolar[3], inducedDipolePolar[3];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
#ifdef HIPPO
    const real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
#endif
    mixed energy = 0;
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        // Compute the torque.

        multipole[1] = labDipole[i*3];
        multipole[2] = labDipole[i*3+1];
        multipole[3] = labDipole[i*3+2];
#ifdef HIPPO
        multipole[0] = coreCharge[i]+valenceCharge[i];
        multipole[4] = labQXX[i];
        multipole[5] = labQYY[i];
        multipole[7] = 2*labQXY[i];
        multipole[8] = 2*labQXZ[i];
        multipole[9] = 2*labQYZ[i];
        const real scale = EPSILON_FACTOR;
#else
        multipole[0] = posq[i].w;
        multipole[4] = labQuadrupole[i*5];
        multipole[5] = labQuadrupole[i*5+3];
        multipole[7] = 2*labQuadrupole[i*5+1];
        multipole[8] = 2*labQuadrupole[i*5+2];
        multipole[9] = 2*labQuadrupole[i*5+4];
        const real scale = EPSILON_FACTOR/2;
#endif
        multipole[6] = -(multipole[4]+multipole[5]);
        GLOBAL const real* cphi = &cphi_global[10*i];
 
        torqueBuffers[i] += realToFixedPoint(scale*(multipole[3]*cphi[2] - multipole[2]*cphi[3]
                      + 2*(multipole[6]-multipole[5])*cphi[9]
                      + multipole[8]*cphi[7] + multipole[9]*cphi[5]
                      - multipole[7]*cphi[8] - multipole[9]*cphi[6]));

        torqueBuffers[i+PADDED_NUM_ATOMS] += realToFixedPoint(scale*(multipole[1]*cphi[3] - multipole[3]*cphi[1]
                      + 2*(multipole[4]-multipole[6])*cphi[8]
                      + multipole[7]*cphi[9] + multipole[8]*cphi[6]
                      - multipole[8]*cphi[4] - multipole[9]*cphi[7]));

        torqueBuffers[i+PADDED_NUM_ATOMS*2] += realToFixedPoint(scale*(multipole[2]*cphi[1] - multipole[1]*cphi[2]
                      + 2*(multipole[5]-multipole[4])*cphi[7]
                      + multipole[7]*cphi[4] + multipole[9]*cphi[8]
                      - multipole[7]*cphi[5] - multipole[8]*cphi[9]));

        // Compute the force and energy.

        multipole[1] = fracDipole[i*3];
        multipole[2] = fracDipole[i*3+1];
        multipole[3] = fracDipole[i*3+2];
        multipole[4] = fracQuadrupole[i*6];
        multipole[5] = fracQuadrupole[i*6+3];
        multipole[6] = fracQuadrupole[i*6+5];
        multipole[7] = fracQuadrupole[i*6+1];
        multipole[8] = fracQuadrupole[i*6+2];
        multipole[9] = fracQuadrupole[i*6+4];

        cinducedDipole[0] = inducedDipole_global[3*i];
        cinducedDipole[1] = inducedDipole_global[3*i+1];
        cinducedDipole[2] = inducedDipole_global[3*i+2];
#ifndef HIPPO
        cinducedDipolePolar[0] = inducedDipolePolar_global[3*i];
        cinducedDipolePolar[1] = inducedDipolePolar_global[3*i+1];
        cinducedDipolePolar[2] = inducedDipolePolar_global[3*i+2];
#endif
        
        // Multiply the dipoles by cartToFrac, which is just the transpose of fracToCart.
        
        inducedDipole[0] = cinducedDipole[0]*fracToCart[0][0] + cinducedDipole[1]*fracToCart[1][0] + cinducedDipole[2]*fracToCart[2][0];
        inducedDipole[1] = cinducedDipole[0]*fracToCart[0][1] + cinducedDipole[1]*fracToCart[1][1] + cinducedDipole[2]*fracToCart[2][1];
        inducedDipole[2] = cinducedDipole[0]*fracToCart[0][2] + cinducedDipole[1]*fracToCart[1][2] + cinducedDipole[2]*fracToCart[2][2];
#ifndef HIPPO
        inducedDipolePolar[0] = cinducedDipolePolar[0]*fracToCart[0][0] + cinducedDipolePolar[1]*fracToCart[1][0] + cinducedDipolePolar[2]*fracToCart[2][0];
        inducedDipolePolar[1] = cinducedDipolePolar[0]*fracToCart[0][1] + cinducedDipolePolar[1]*fracToCart[1][1] + cinducedDipolePolar[2]*fracToCart[2][1];
        inducedDipolePolar[2] = cinducedDipolePolar[0]*fracToCart[0][2] + cinducedDipolePolar[1]*fracToCart[1][2] + cinducedDipolePolar[2]*fracToCart[2][2];
        energy += (inducedDipole[0]+inducedDipolePolar[0])*phi[i+NUM_ATOMS];
        energy += (inducedDipole[1]+inducedDipolePolar[1])*phi[i+NUM_ATOMS*2];
        energy += (inducedDipole[2]+inducedDipolePolar[2])*phi[i+NUM_ATOMS*3];
#endif
        real3 f = make_real3(0, 0, 0);
        for (int k = 0; k < 3; k++) {
            int j1 = deriv1[k+1];
            int j2 = deriv2[k+1];
            int j3 = deriv3[k+1];
#ifdef HIPPO
            f.x += inducedDipole[k]*phi[i+NUM_ATOMS*j1];
            f.y += inducedDipole[k]*phi[i+NUM_ATOMS*j2];
            f.z += inducedDipole[k]*phi[i+NUM_ATOMS*j3];
#else
            f.x += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j1];
            f.y += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j2];
            f.z += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j3];
#endif
#ifdef MUTUAL_POLARIZATION
            f.x += (inducedDipole[k]*phip[i+NUM_ATOMS*j1] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j1]);
            f.y += (inducedDipole[k]*phip[i+NUM_ATOMS*j2] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j2]);
            f.z += (inducedDipole[k]*phip[i+NUM_ATOMS*j3] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j3]);
#endif
        }

        for (int k = 0; k < 10; k++) {
            f.x += multipole[k]*phidp[i+NUM_ATOMS*deriv1[k]];
            f.y += multipole[k]*phidp[i+NUM_ATOMS*deriv2[k]];
            f.z += multipole[k]*phidp[i+NUM_ATOMS*deriv3[k]];
        }

#ifdef HIPPO
        // Account for dipole response terms in the OPT method

        for (int j = 0; j < MAX_EXTRAPOLATION_ORDER-1; j++) {
            for (int m = 0; m < MAX_EXTRAPOLATION_ORDER-1-j; m++) {
                real3 optDipole = make_real3(
                        extrapolatedDipole[3*NUM_ATOMS*m+3*i]*fracToCart[0][0] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+1]*fracToCart[1][0] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+2]*fracToCart[2][0],
                        extrapolatedDipole[3*NUM_ATOMS*m+3*i]*fracToCart[0][1] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+1]*fracToCart[1][1] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+2]*fracToCart[2][1],
                        extrapolatedDipole[3*NUM_ATOMS*m+3*i]*fracToCart[0][2] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+1]*fracToCart[1][2] + extrapolatedDipole[3*NUM_ATOMS*m+3*i+2]*fracToCart[2][2]);
                real3 h = make_real3(
                        optDipole.x*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv1[1]] + optDipole.y*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv1[2]] + optDipole.z*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv1[3]],
                        optDipole.x*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv2[1]] + optDipole.y*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv2[2]] + optDipole.z*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv2[3]],
                        optDipole.x*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv3[1]] + optDipole.y*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv3[2]] + optDipole.z*extrapolatedPhi[10*NUM_ATOMS*j+i+NUM_ATOMS*deriv3[3]]);
                f += coeff[j+m+1]*h;
            }
        }
#endif
        f = make_real3(scale*(f.x*fracToCart[0][0] + f.y*fracToCart[0][1] + f.z*fracToCart[0][2]),
                       scale*(f.x*fracToCart[1][0] + f.y*fracToCart[1][1] + f.z*fracToCart[1][2]),
                       scale*(f.x*fracToCart[2][0] + f.y*fracToCart[2][1] + f.z*fracToCart[2][2]));
        forceBuffers[i] -= (mm_ulong) realToFixedPoint(f.x);
        forceBuffers[i+PADDED_NUM_ATOMS] -= (mm_ulong) realToFixedPoint(f.y);
        forceBuffers[i+PADDED_NUM_ATOMS*2] -= (mm_ulong) realToFixedPoint(f.z);
    }
#ifndef HIPPO
    energyBuffer[GLOBAL_ID] += 0.25f*EPSILON_FACTOR*energy;
#endif
}

#ifdef HIPPO
KERNEL void recordInducedFieldDipoles(GLOBAL const real* RESTRICT phidp, GLOBAL mm_long* RESTRICT inducedField,
        GLOBAL const real* RESTRICT inducedDipole, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    real selfDipoleScale = (4/(real) 3)*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/SQRT_PI;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        inducedField[i] -= realToFixedPoint(phidp[i+NUM_ATOMS]*fracToCart[0][0] + phidp[i+NUM_ATOMS*2]*fracToCart[0][1] + phidp[i+NUM_ATOMS*3]*fracToCart[0][2] - selfDipoleScale*inducedDipole[3*i]);
        inducedField[i+PADDED_NUM_ATOMS] -= realToFixedPoint(phidp[i+NUM_ATOMS]*fracToCart[1][0] + phidp[i+NUM_ATOMS*2]*fracToCart[1][1] + phidp[i+NUM_ATOMS*3]*fracToCart[1][2] - selfDipoleScale*inducedDipole[3*i+1]);
        inducedField[i+PADDED_NUM_ATOMS*2] -= realToFixedPoint(phidp[i+NUM_ATOMS]*fracToCart[2][0] + phidp[i+NUM_ATOMS*2]*fracToCart[2][1] + phidp[i+NUM_ATOMS*3]*fracToCart[2][2] - selfDipoleScale*inducedDipole[3*i+2]);
    }
}

KERNEL void calculateSelfEnergyAndTorque(GLOBAL mm_long* RESTRICT torqueBuffers, GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real* RESTRICT labDipole, GLOBAL const real* RESTRICT coreCharge, GLOBAL const real* RESTRICT valenceCharge,
        GLOBAL const real* RESTRICT c6, GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT labQXX, GLOBAL const real* RESTRICT labQXY,
        GLOBAL const real* RESTRICT labQXZ, GLOBAL const real* RESTRICT labQYY, GLOBAL const real* RESTRICT labQYZ) {
    const real torqueScale = 4*EPSILON_FACTOR*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/(3*SQRT_PI);
    real cii = 0;
    real dii = 0;
    real qii = 0;
    real c6ii = 0;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        real charge = coreCharge[i]+valenceCharge[i];
        real3 dipole = make_real3(labDipole[3*i], labDipole[3*i+1], labDipole[3*i+2]);
        real3 induced = make_real3(inducedDipole[3*i], inducedDipole[3*i+1], inducedDipole[3*i+2]);
        real qXX = labQXX[i];
        real qXY = labQXY[i];
        real qXZ = labQXZ[i];
        real qYY = labQYY[i];
        real qYZ = labQYZ[i];
        real qZZ = -qXX-qYY;
        real c6i = c6[i];
        cii += charge*charge;
        dii += dot(dipole, dipole);
        qii += qXX*qXX + qYY*qYY + qZZ*qZZ + 2*(qXY*qXY + qXZ*qXZ + qYZ*qYZ);
        c6ii += c6i*c6i;
        real3 torque = torqueScale*cross(dipole, induced);
        torqueBuffers[i] += realToFixedPoint(torque.x);
        torqueBuffers[i+PADDED_NUM_ATOMS] += realToFixedPoint(torque.y);
        torqueBuffers[i+PADDED_NUM_ATOMS*2] += realToFixedPoint(torque.z);
    }
    real term = 2*EWALD_ALPHA*EWALD_ALPHA;
    real fterm = -EPSILON_FACTOR*EWALD_ALPHA/SQRT_PI;
    real alpha3 = DISPERSION_EWALD_ALPHA*DISPERSION_EWALD_ALPHA*DISPERSION_EWALD_ALPHA;
    energyBuffer[GLOBAL_ID] += fterm*(cii + term*(dii/3+2*term*qii/5)) + alpha3*alpha3*c6ii/12;
}
#else
KERNEL void recordInducedFieldDipoles(GLOBAL const real* RESTRICT phid, GLOBAL real* const RESTRICT phip, GLOBAL mm_long* RESTRICT inducedField,
        GLOBAL mm_long* RESTRICT inducedFieldPolar, GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ
#ifdef EXTRAPOLATED_POLARIZATION
        , GLOBAL mm_ulong* RESTRICT fieldGradient, GLOBAL mm_ulong* RESTRICT fieldGradientPolar
#endif
        ) {
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    real selfDipoleScale = (4/(real) 3)*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/SQRT_PI;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        inducedField[i] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[0][0] + phid[i+NUM_ATOMS*2]*fracToCart[0][1] + phid[i+NUM_ATOMS*3]*fracToCart[0][2] - selfDipoleScale*inducedDipole[3*i]);
        inducedField[i+PADDED_NUM_ATOMS] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[1][0] + phid[i+NUM_ATOMS*2]*fracToCart[1][1] + phid[i+NUM_ATOMS*3]*fracToCart[1][2] - selfDipoleScale*inducedDipole[3*i+1]);
        inducedField[i+PADDED_NUM_ATOMS*2] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[2][0] + phid[i+NUM_ATOMS*2]*fracToCart[2][1] + phid[i+NUM_ATOMS*3]*fracToCart[2][2] - selfDipoleScale*inducedDipole[3*i+2]);
        inducedFieldPolar[i] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[0][0] + phip[i+NUM_ATOMS*2]*fracToCart[0][1] + phip[i+NUM_ATOMS*3]*fracToCart[0][2] - selfDipoleScale*inducedDipolePolar[3*i]);
        inducedFieldPolar[i+PADDED_NUM_ATOMS] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[1][0] + phip[i+NUM_ATOMS*2]*fracToCart[1][1] + phip[i+NUM_ATOMS*3]*fracToCart[1][2] - selfDipoleScale*inducedDipolePolar[3*i+1]);
        inducedFieldPolar[i+PADDED_NUM_ATOMS*2] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[2][0] + phip[i+NUM_ATOMS*2]*fracToCart[2][1] + phip[i+NUM_ATOMS*3]*fracToCart[2][2] - selfDipoleScale*inducedDipolePolar[3*i+2]);
#ifdef EXTRAPOLATED_POLARIZATION
        // Compute and store the field gradients for later use.

        real EmatD[3][3] = {
            {phid[i+NUM_ATOMS*4], phid[i+NUM_ATOMS*7], phid[i+NUM_ATOMS*8]},
            {phid[i+NUM_ATOMS*7], phid[i+NUM_ATOMS*5], phid[i+NUM_ATOMS*9]},
            {phid[i+NUM_ATOMS*8], phid[i+NUM_ATOMS*9], phid[i+NUM_ATOMS*6]}
        };
        real Exx = 0, Eyy = 0, Ezz = 0, Exy = 0, Exz = 0, Eyz = 0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Exx += fracToCart[0][k] * EmatD[k][l] * fracToCart[0][l];
                Eyy += fracToCart[1][k] * EmatD[k][l] * fracToCart[1][l];
                Ezz += fracToCart[2][k] * EmatD[k][l] * fracToCart[2][l];
                Exy += fracToCart[0][k] * EmatD[k][l] * fracToCart[1][l];
                Exz += fracToCart[0][k] * EmatD[k][l] * fracToCart[2][l];
                Eyz += fracToCart[1][k] * EmatD[k][l] * fracToCart[2][l];
            }
        }
        ATOMIC_ADD(&fieldGradient[6*i+0], (mm_ulong) realToFixedPoint(-Exx));
        ATOMIC_ADD(&fieldGradient[6*i+1], (mm_ulong) realToFixedPoint(-Eyy));
        ATOMIC_ADD(&fieldGradient[6*i+2], (mm_ulong) realToFixedPoint(-Ezz));
        ATOMIC_ADD(&fieldGradient[6*i+3], (mm_ulong) realToFixedPoint(-Exy));
        ATOMIC_ADD(&fieldGradient[6*i+4], (mm_ulong) realToFixedPoint(-Exz));
        ATOMIC_ADD(&fieldGradient[6*i+5], (mm_ulong) realToFixedPoint(-Eyz));

        real EmatP[3][3] = {
            {phip[i+NUM_ATOMS*4], phip[i+NUM_ATOMS*7], phip[i+NUM_ATOMS*8]},
            {phip[i+NUM_ATOMS*7], phip[i+NUM_ATOMS*5], phip[i+NUM_ATOMS*9]},
            {phip[i+NUM_ATOMS*8], phip[i+NUM_ATOMS*9], phip[i+NUM_ATOMS*6]}
        };
        Exx = 0; Eyy = 0; Ezz = 0; Exy = 0; Exz = 0; Eyz = 0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Exx += fracToCart[0][k] * EmatP[k][l] * fracToCart[0][l];
                Eyy += fracToCart[1][k] * EmatP[k][l] * fracToCart[1][l];
                Ezz += fracToCart[2][k] * EmatP[k][l] * fracToCart[2][l];
                Exy += fracToCart[0][k] * EmatP[k][l] * fracToCart[1][l];
                Exz += fracToCart[0][k] * EmatP[k][l] * fracToCart[2][l];
                Eyz += fracToCart[1][k] * EmatP[k][l] * fracToCart[2][l];
            }
        }
        ATOMIC_ADD(&fieldGradientPolar[6*i+0], (mm_ulong) realToFixedPoint(-Exx));
        ATOMIC_ADD(&fieldGradientPolar[6*i+1], (mm_ulong) realToFixedPoint(-Eyy));
        ATOMIC_ADD(&fieldGradientPolar[6*i+2], (mm_ulong) realToFixedPoint(-Ezz));
        ATOMIC_ADD(&fieldGradientPolar[6*i+3], (mm_ulong) realToFixedPoint(-Exy));
        ATOMIC_ADD(&fieldGradientPolar[6*i+4], (mm_ulong) realToFixedPoint(-Exz));
        ATOMIC_ADD(&fieldGradientPolar[6*i+5], (mm_ulong) realToFixedPoint(-Eyz));
#endif
    }
}
#endif
