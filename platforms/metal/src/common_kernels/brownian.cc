/**
 * Perform the first step of Brownian integration.
 */

KERNEL void integrateBrownianPart1(int numAtoms, int paddedNumAtoms, mixed tauDeltaT, mixed noiseAmplitude, GLOBAL const mm_long* RESTRICT force,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL const mixed4* RESTRICT velm, GLOBAL const float4* RESTRICT random, unsigned int randomIndex) {
    randomIndex += GLOBAL_ID;
#ifdef USE_DOUBLE_SINGLE
    const mixed fscale = DS_div(tauDeltaT, DS_init(0x100000000, 0));
#else
    const mixed fscale = tauDeltaT/(mixed) 0x100000000;
#endif
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed invMass = velm[index].w;
        if (invMass != 0) {
#ifdef USE_DOUBLE_SINGLE
            mixed temp1 = DS_mul(fscale, invMass);
            mixed temp2 = DS_mul(noiseAmplitude, DS_sqrt(invMass));
            float4 in = random[randomIndex];
            mixed3 out;
            out.x = DS_add(DS_mul(temp1, DS_init_long(force[index])), DS_mul_float_rhs(temp2, in.x));
            out.y = DS_add(DS_mul(temp1, DS_init_long(force[index+paddedNumAtoms])), DS_mul_float_rhs(temp2, in.y));
            out.z = DS_add(DS_mul(temp1, DS_init_long(force[index+paddedNumAtoms*2])), DS_mul_float_rhs(temp2, in.z));
            posDelta[index] = out;
#else
            posDelta[index].x = fscale*invMass*force[index] + noiseAmplitude*SQRT(invMass)*random[randomIndex].x;
            posDelta[index].y = fscale*invMass*force[index+paddedNumAtoms] + noiseAmplitude*SQRT(invMass)*random[randomIndex].y;
            posDelta[index].z = fscale*invMass*force[index+paddedNumAtoms*2] + noiseAmplitude*SQRT(invMass)*random[randomIndex].z;
#endif
        }
        randomIndex += GLOBAL_SIZE;
    }
}

/**
 * Perform the second step of Brownian integration.
 */

KERNEL void integrateBrownianPart2(int numAtoms, mixed oneOverDeltaT, GLOBAL real4* posq, GLOBAL mixed4* velm, GLOBAL const mixed4* RESTRICT posDelta
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
        ) {
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        if (velm[index].w != 0) {
            mixed4 delta = posDelta[index];
#if USE_DOUBLE_SINGLE
            velm[index].x = DS_mul(oneOverDeltaT, delta.x);
            velm[index].y = DS_mul(oneOverDeltaT, delta.y);
            velm[index].z = DS_mul(oneOverDeltaT, delta.z);
            
            // Never assume positions are correctly normalized.
            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = DS4_init(DS_init_adding(pos1.x, pos2.x),
                                  DS_init_adding(pos1.y, pos2.y),
                                  DS_init_adding(pos1.z, pos2.z),
                                  DS_init(pos1.w, 0));
            
            pos.x = DS_add(pos.x, delta.x);
            pos.y = DS_add(pos.y, delta.y);
            pos.z = DS_add(pos.y, delta.z);
            posq[index] = make_real4(pos.x.hi, pos.y.hi, pos.z.hi, pos.w.hi);
            posqCorrection[index] = make_real4(pos.x.lo, pos.y.lo, pos.z.lo, 0);
#else
            velm[index].x = oneOverDeltaT*delta.x;
            velm[index].y = oneOverDeltaT*delta.y;
            velm[index].z = oneOverDeltaT*delta.z;
    #ifdef USE_MIXED_PRECISION
            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
    #else
            real4 pos = posq[index];
    #endif
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
    #ifdef USE_MIXED_PRECISION
            posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
            posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
    #else
            posq[index] = pos;
    #endif
#endif
        }
    }
}
