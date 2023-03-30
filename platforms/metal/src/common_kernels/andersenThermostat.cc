/**
 * Apply the Andersen thermostat to adjust particle velocities.
 */

KERNEL void applyAndersenThermostat(int numAtoms, float collisionFrequency, float kT, GLOBAL mixed4* velm, real stepSize, GLOBAL const float4* RESTRICT random,
        unsigned int randomIndex, GLOBAL const int* RESTRICT atomGroups) {
    float collisionProbability = (float) (1-EXP(-collisionFrequency*stepSize));
    float randomRange = (float) erf(collisionProbability/SQRT(2.0f));
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        float4 selectRand = random[randomIndex+atomGroups[index]];
        float4 velRand = random[randomIndex+index];
        real scale = (selectRand.w > -randomRange && selectRand.w < randomRange ? 0 : 1);
#ifdef USE_DOUBLE_SINGLE
        real add = (1-scale)*SQRT(kT*velocity.w.hi);
        velocity.x = DS_add_float_rhs(DS_mul_float_lhs(scale, velocity.x), add*velRand.x);
        velocity.y = DS_add_float_rhs(DS_mul_float_lhs(scale, velocity.y), add*velRand.y);
        velocity.z = DS_add_float_rhs(DS_mul_float_lhs(scale, velocity.z), add*velRand.z);
#else
        real add = (1-scale)*SQRT(kT*velocity.w);
        velocity.x = scale*velocity.x + add*velRand.x;
        velocity.y = scale*velocity.y + add*velRand.y;
        velocity.z = scale*velocity.z + add*velRand.z;
#endif
        velm[index] = velocity;
    }
}
