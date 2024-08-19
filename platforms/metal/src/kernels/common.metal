/**
 * This file contains Metal definitions for the macros and functions needed for the
 * common compute framework.
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
__attribute__((overloadable)) unsigned long atom_add(volatile __global unsigned long* p, unsigned long val) {
    volatile __global unsigned int* word = (volatile __global unsigned int*) p;
#ifdef __ENDIAN_LITTLE__
    int lowIndex = 0;
#else
    int lowIndex = 1;
#endif
    unsigned int lower = val;
    unsigned int upper = val >> 32;
    unsigned int result = atomic_add(&word[lowIndex], lower);
    int carry = (lower + (unsigned long) result >= 0x100000000 ? 1 : 0);
    upper += carry;
    if (upper != 0)
        atomic_add(&word[1-lowIndex], upper);
    return 0;
}
#endif

#define KERNEL __kernel
#define DEVICE
#define LOCAL __local
#define LOCAL_ARG __local
#define GLOBAL __global
#define RESTRICT restrict
#define LOCAL_ID get_local_id(0)
#define LOCAL_SIZE get_local_size(0)
#define GLOBAL_ID get_global_id(0)
#define GLOBAL_SIZE get_global_size(0)
#define GROUP_ID get_group_id(0)
#define NUM_GROUPS get_num_groups(0)
#define SYNC_THREADS barrier(CLK_LOCAL_MEM_FENCE+CLK_GLOBAL_MEM_FENCE);
#define MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE+CLK_GLOBAL_MEM_FENCE);
#define ATOMIC_ADD(dest, value) atom_add(dest, value)

#ifdef VENDOR_APPLE

#define FORCE_UNROLL_4(__expr, __start_index) \
__expr(__start_index + 0); \
__expr(__start_index + 1); \
__expr(__start_index + 2); \
__expr(__start_index + 3); \

#define FORCE_UNROLL_32(__expr) \
FORCE_UNROLL_4(__expr, 0); \
FORCE_UNROLL_4(__expr, 4); \
FORCE_UNROLL_4(__expr, 8); \
FORCE_UNROLL_4(__expr, 12); \
FORCE_UNROLL_4(__expr, 16); \
FORCE_UNROLL_4(__expr, 20); \
FORCE_UNROLL_4(__expr, 24); \
FORCE_UNROLL_4(__expr, 28); \

#endif // VENDOR_APPLE

typedef long mm_long;
typedef unsigned long mm_ulong;

#define make_short2(x...) ((short2) (x))
#define make_short3(x...) ((short3) (x))
#define make_short4(x...) ((short4) (x))
#define make_int2(x...) ((int2) (x))
#define make_int3(x...) ((int3) (x))
#define make_int4(x...) ((int4) (x))
#define make_float2(x...) ((float2) (x))
#define make_float3(x...) ((float3) (x))
#define make_float4(x...) ((float4) (x))
#define make_double2(x...) ((double2) (x))
#define make_double3(x...) ((double3) (x))
#define make_double4(x...) ((double4) (x))

#define trimTo3(v) (v).xyz

// Metal has overloaded versions of standard math functions for single and double
// precision arguments.  CUDA has separate functions.  To allow them to be called
// consistently, we define the "single precision" functions to just be synonyms
// for the standard ones.

#define sqrtf(x) sqrt(x)
#define rsqrtf(x) rsqrt(x)
#define expf(x) exp(x)
#define logf(x) log(x)
#define powf(x) pow(x)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define tanf(x) tan(x)
#define acosf(x) acos(x)
#define asinf(x) asin(x)
#define atanf(x) atan(x)
#define atan2f(x, y) atan2(x, y)

inline long realToFixedPoint(real x) {
    return (long) (x * 0x100000000);
}



#ifdef VENDOR_APPLE

// Count trailing zeroes instruction (Int32)

__attribute__((__overloadable__)) 
int ___metal_ctz(int data, bool unknown)
  __asm("air.ctz.i32");

__attribute__((__overloadable__)) 
int ctz(int x) {
  return ___metal_ctz(x, false);
}

// SIMD sum instruction (Float32)

__attribute__((__overloadable__))
float simd_sum(float data)
__asm("air.simd_sum.f32");

__attribute__((__overloadable__))
float sub_group_reduce_add(float x) {
  return simd_sum(x);
}

// SIMD ballot instruction (Int32)

#define EXPOSE_BALLOT(FUNC_EXPR, IN_EXPR, OUT_EXPR, AIR_EXPR) \
__attribute__((__overloadable__)) OUT_EXPR FUNC_EXPR(IN_EXPR) \
  __asm("air." #FUNC_EXPR #AIR_EXPR); \

EXPOSE_BALLOT(simd_is_first, , bool, )
EXPOSE_BALLOT(simd_all, bool expr, bool, )
EXPOSE_BALLOT(simd_any, bool expr, bool, )
EXPOSE_BALLOT(simd_ballot, bool expr, ulong, .i64)
EXPOSE_BALLOT(simd_active_threads_mask, , ulong, .i64)

int sub_group_elect() {
  return select(0, 1, simd_is_first());
}

int sub_group_all(int predicate) {
  return select(0, 1, simd_all(predicate != 0));
}

int sub_group_any(int predicate) {
  return select(0, 1, simd_any(predicate != 0));
}

int sub_group_non_uniform_all(int predicate) {
  return select(0, 1, simd_all(predicate != 0));
}

int sub_group_non_uniform_any(int predicate) {
  return select(0, 1, simd_any(predicate != 0));
}

uint4 sub_group_ballot(int predicate) {
  uint4 output = uint4(0);
  output.x = simd_ballot(predicate != 0);
  return output;
}

#endif // VENDOR_APPLE
