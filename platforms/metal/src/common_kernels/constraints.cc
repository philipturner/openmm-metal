KERNEL void applyPositionDeltas(int numAtoms, GLOBAL real4* RESTRICT posq, GLOBAL mixed4* RESTRICT posDelta
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
        ) {
    for (unsigned int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
#ifdef USE_DOUBLE_SINGLE
        // Never assume positions are correctly normalized.
        real4 pos1 = posq[index];
        real4 pos2 = posqCorrection[index];
        mixed4 pos = DS4_init(DS_init_adding(pos1.x, pos2.x),
                              DS_init_adding(pos1.y, pos2.y),
                              DS_init_adding(pos1.z, pos2.z),
                              DS_init(pos1.w, 0));
        
        mixed3 delta = *((GLOBAL mixed3*)(posDelta + index));
        pos.x = DS_add(pos.x, delta.x);
        pos.y = DS_add(pos.y, delta.y);
        pos.z = DS_add(pos.y, delta.z);
        posq[index] = make_real4(pos.x.hi, pos.y.hi, pos.z.hi, pos.w.hi);
        posqCorrection[index] = make_real4(pos.x.lo, pos.y.lo, pos.z.lo, 0);
#else
    #ifdef USE_MIXED_PRECISION
        real4 pos1 = posq[index];
        real4 pos2 = posqCorrection[index];
        mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
    #else
        mixed4 pos = posq[index];
    #endif
        pos.x += posDelta[index].x;
        pos.y += posDelta[index].y;
        pos.z += posDelta[index].z;
    #ifdef USE_MIXED_PRECISION
        posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
        posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
    #else
        posq[index] = pos;
    #endif
#endif
    }
}
