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

//
//  metal_stdlib.h
//  OpenCL Metal Stdlib
//
//  Created by Philip Turner on 2/26/23.
//

#ifndef metal_stdlib_h
#define metal_stdlib_h

#define EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, C_TYPE, AIR_TYPE) \
__attribute__((__overloadable__)) C_TYPE EXPR(C_TYPE data) \
  __asm("air." #EXPR "." #AIR_TYPE); \

#define EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, C_TYPE, AIR_TYPE) \
__attribute__((__overloadable__)) C_TYPE EXPR(C_TYPE data, ushort delta) \
  __asm("air." #EXPR "." #AIR_TYPE); \

#define EXPOSE_FUNCTION_OVERLOAD_ARGS_3(EXPR, C_TYPE, AIR_TYPE) \
__attribute__((__overloadable__)) C_TYPE ___metal_##EXPR(C_TYPE data, bool unknown) \
  __asm("air." #EXPR "." #AIR_TYPE); \
\
__attribute__((__overloadable__)) C_TYPE EXPR(C_TYPE x) { \
  return ___metal_##EXPR(x, false); \
} \

#define BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x) { \
  return METAL_EXPR(x); \
} \

#define BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, uint delta) { \
  return METAL_EXPR(x, ushort(delta)); \
} \

#define OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  if (clustersize == 4) { \
    return quad_##METAL_SUFFIX(x); \
  } else { \
    return simd_##METAL_SUFFIX(x); \
  } \
} \

#define OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, C_TYPE) \
__attribute__((__overloadable__)) \
C_TYPE OPENCL_EXPR(C_TYPE x, int delta, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  if (clustersize == 4) { \
    return quad_##METAL_SUFFIX(x, ushort(delta)); \
  } else { \
    return simd_##METAL_SUFFIX(x, ushort(delta)); \
  } \
} \

// Removing subgroup extended types to reduce startup overhead.
// TODO: Remove other unused functions too.

#define EXPOSE_FUNCTION_I_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, uint, u.i32) \

#define EXPOSE_FUNCTION_F_ARGS_1(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_1(EXPR, float, f32) \

#define EXPOSE_FUNCTION_ARGS_2(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, int, s.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, uint, u.i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_2(EXPR, float, f32) \

#define EXPOSE_FUNCTION_I_ARGS_3(EXPR) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_3(EXPR, int, i32) \
EXPOSE_FUNCTION_OVERLOAD_ARGS_3(EXPR, uint, i32) \

#define BRIDGE_FUNCTION_I_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, uint) \

#define BRIDGE_FUNCTION_F_ARGS_1(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_1(METAL_EXPR, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_ARGS_2(METAL_EXPR, OPENCL_EXPR) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, int) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, uint) \
BRIDGE_FUNCTION_OVERLOAD_ARGS_2(METAL_EXPR, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, uint) \

#define BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_1(METAL_SUFFIX, OPENCL_EXPR, float) \

#define BRIDGE_FUNCTION_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, int) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, uint) \
OVERLOAD_CLUSTERED_ARGS_2(METAL_SUFFIX, OPENCL_EXPR, float) \

// Declarations

// `I` = integer reduction
// `F` = floating-point reduction
// `B` = boolean reduction

#define DECLARE_I_REDUCTION_BASE(METAL_OP) \
EXPOSE_FUNCTION_I_ARGS_1(quad_##METAL_OP) \
EXPOSE_FUNCTION_I_ARGS_1(simd_##METAL_OP) \

#define DECLARE_F_REDUCTION_BASE(METAL_OP) \
EXPOSE_FUNCTION_F_ARGS_1(quad_##METAL_OP) \
EXPOSE_FUNCTION_F_ARGS_1(simd_##METAL_OP) \

#define DECLARE_REDUCTION_BASE(METAL_OP) \
DECLARE_I_REDUCTION_BASE(METAL_OP) \
DECLARE_F_REDUCTION_BASE(METAL_OP) \

#define DECLARE_SHUFFLE_BASE(METAL_OP) \
EXPOSE_FUNCTION_ARGS_2(quad_##METAL_OP) \
EXPOSE_FUNCTION_ARGS_2(simd_##METAL_OP) \

DECLARE_REDUCTION_BASE(sum)
DECLARE_REDUCTION_BASE(prefix_inclusive_sum)
DECLARE_REDUCTION_BASE(prefix_exclusive_sum)
DECLARE_REDUCTION_BASE(min)
DECLARE_REDUCTION_BASE(max)

DECLARE_REDUCTION_BASE(product)
DECLARE_REDUCTION_BASE(prefix_inclusive_product)
DECLARE_REDUCTION_BASE(prefix_exclusive_product)
DECLARE_I_REDUCTION_BASE(and)
DECLARE_I_REDUCTION_BASE(or)
DECLARE_I_REDUCTION_BASE(xor)

DECLARE_SHUFFLE_BASE(broadcast)
DECLARE_REDUCTION_BASE(broadcast_first)

DECLARE_SHUFFLE_BASE(shuffle)
DECLARE_SHUFFLE_BASE(shuffle_xor)
DECLARE_SHUFFLE_BASE(shuffle_up)
DECLARE_SHUFFLE_BASE(shuffle_down)
DECLARE_SHUFFLE_BASE(shuffle_rotate_up)
DECLARE_SHUFFLE_BASE(shuffle_rotate_down)

EXPOSE_FUNCTION_I_ARGS_3(ctz)

#define DECLARE_I_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_I_ARGS_1(simd_##METAL_OP, sub_group_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_F_ARGS_1(simd_##METAL_OP, sub_group_##OPENCL_OP) \

#define DECLARE_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_UNIFORM(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_ARGS_2(simd_##METAL_OP, sub_group_##OPENCL_OP) \

DECLARE_REDUCTION_UNIFORM(sum, reduce_add)
DECLARE_REDUCTION_UNIFORM(prefix_inclusive_sum, scan_inclusive_add)
DECLARE_REDUCTION_UNIFORM(prefix_exclusive_sum, scan_exclusive_add)
DECLARE_REDUCTION_UNIFORM(min, reduce_min)
DECLARE_REDUCTION_UNIFORM(max, reduce_max)

DECLARE_SHUFFLE_UNIFORM(shuffle, shuffle)
DECLARE_SHUFFLE_UNIFORM(shuffle_xor, shuffle_xor)
DECLARE_SHUFFLE_UNIFORM(shuffle_up, shuffle_up)
DECLARE_SHUFFLE_UNIFORM(shuffle_down, shuffle_down)
DECLARE_SHUFFLE_UNIFORM(shuffle_rotate_down, rotate)

DECLARE_SHUFFLE_UNIFORM(broadcast, broadcast)
DECLARE_REDUCTION_UNIFORM(broadcast_first, broadcast_first)

#define DECLARE_I_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_I_ARGS_1(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_F_ARGS_1(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

#define DECLARE_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_NON_UNIFORM(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_NON_UNIFORM(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_ARGS_2(simd_##METAL_OP, sub_group_non_uniform_##OPENCL_OP) \

DECLARE_REDUCTION_NON_UNIFORM(sum, reduce_add)
DECLARE_REDUCTION_NON_UNIFORM(prefix_inclusive_sum, scan_inclusive_add)
DECLARE_REDUCTION_NON_UNIFORM(prefix_exclusive_sum, scan_exclusive_add)
DECLARE_REDUCTION_NON_UNIFORM(min, reduce_min)
DECLARE_REDUCTION_NON_UNIFORM(max, reduce_max)

DECLARE_REDUCTION_NON_UNIFORM(product, reduce_mul)
DECLARE_REDUCTION_NON_UNIFORM(prefix_inclusive_product, scan_inclusive_mul)
DECLARE_REDUCTION_NON_UNIFORM(prefix_exclusive_product, scan_exclusive_mul)
DECLARE_I_REDUCTION_NON_UNIFORM(and, reduce_and)
DECLARE_I_REDUCTION_NON_UNIFORM(or, reduce_or)
DECLARE_I_REDUCTION_NON_UNIFORM(xor, reduce_xor)

DECLARE_SHUFFLE_NON_UNIFORM(broadcast, broadcast)

#define DECLARE_I_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1(METAL_OP, sub_group_clustered_##OPENCL_OP) \

#define DECLARE_F_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1(METAL_OP, sub_group_clustered_##OPENCL_OP) \

#define DECLARE_B_REDUCTION_CLUSTERED(OP) \
__attribute__((__overloadable__)) \
int sub_group_non_uniform_reduce_logical_##OP(int predicate) { \
return simd_##OP(select(0, 1, predicate != 0)); \
} \
\
__attribute__((__overloadable__)) \
int sub_group_clustered_reduce_logical_##OP(int predicate, uint clustersize) \
  __attribute__((enable_if(clustersize == 4 || clustersize == 32, "Cluster size not supported."))) \
{ \
  int x = select(0, 1, predicate != 0); \
  if (clustersize == 4) { \
    return quad_##OP(x); \
  } else { \
    return simd_##OP(x); \
  } \
} \

#define DECLARE_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
DECLARE_I_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \
DECLARE_F_REDUCTION_CLUSTERED(METAL_OP, OPENCL_OP) \

#define DECLARE_SHUFFLE_CLUSTERED(METAL_OP, OPENCL_OP) \
BRIDGE_FUNCTION_CLUSTERED_ARGS_2(METAL_OP, sub_group_clustered_##OPENCL_OP) \

DECLARE_REDUCTION_CLUSTERED(sum, reduce_add)
DECLARE_REDUCTION_CLUSTERED(min, reduce_min)
DECLARE_REDUCTION_CLUSTERED(max, reduce_max)

DECLARE_REDUCTION_CLUSTERED(product, reduce_mul)
DECLARE_I_REDUCTION_CLUSTERED(and, reduce_and)
DECLARE_I_REDUCTION_CLUSTERED(or, reduce_or)
DECLARE_I_REDUCTION_CLUSTERED(xor, reduce_xor)
DECLARE_B_REDUCTION_CLUSTERED(and)
DECLARE_B_REDUCTION_CLUSTERED(or)
DECLARE_B_REDUCTION_CLUSTERED(xor)

DECLARE_SHUFFLE_CLUSTERED(shuffle_rotate_down, rotate)

#define EXPOSE_BALLOT(FUNC_EXPR, IN_EXPR, OUT_EXPR, AIR_EXPR) \
__attribute__((__overloadable__)) OUT_EXPR FUNC_EXPR(IN_EXPR) \
  __asm("air." #FUNC_EXPR #AIR_EXPR); \

EXPOSE_BALLOT(simd_is_first, , bool, )
EXPOSE_BALLOT(simd_all, bool expr, bool, )
EXPOSE_BALLOT(simd_any, bool expr, bool, )
EXPOSE_BALLOT(simd_ballot, bool expr, ulong, .i64)
EXPOSE_BALLOT(simd_active_threads_mask, , ulong, .i64)

EXPOSE_BALLOT(quad_is_first, , bool, )
EXPOSE_BALLOT(quad_all, bool expr, bool, )
EXPOSE_BALLOT(quad_any, bool expr, bool, )
EXPOSE_BALLOT(quad_ballot, bool expr, ushort, )
EXPOSE_BALLOT(quad_active_threads_mask, , ushort, )

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

// WARNING: For SIMD-scoped barriers, Metal and OpenCL have different
// enumeration constants for the memory flags.

// Metal:
// - scope threadgroup = 2
// - scope device = 1
// - scope texture = 4

#define __METAL_MEMORY_FLAGS_NONE__ 0
#define __METAL_MEMORY_FLAGS_DEVICE__ 1
#define __METAL_MEMORY_FLAGS_THREADGROUP__ 2
#define __METAL_MEMORY_FLAGS_TEXTURE__ 4
#define __METAL_MEMORY_FLAGS_THREADGROUP_IMAGEBLOCK__ 8

// OpenCL:
// - scope local = 1
// - scope global = 2
// - scope image = 4

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE 1
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE 2
#endif

#ifndef CLK_IMAGE_MEM_FENCE
#define CLK_IMAGE_MEM_FENCE 4
#endif

__attribute__((__overloadable__))
void __metal_simdgroup_barrier(int flags, int scope)
  __asm("air.simdgroup.barrier");

__attribute__((__overloadable__))
void simdgroup_barrier(int flags) {
  // __METAL_MEMORY_SCOPE_THREADGROUP__ = 1
  // The same scope is also used for `__metal_wg_barrier`, while
  // `__METAL_MEMORY_SCOPE_SIMDGROUP__` is not used for either.
  __metal_simdgroup_barrier(flags, 1);
}

__attribute__((__overloadable__))
__attribute__((__always_inline__)) // Try to force constant evaluation.
// NOTE: Always profile against the raw Metal binding, as we cannot guarantee
// the compiler will perform constant evaluation.
void sub_group_barrier(cl_mem_fence_flags flags)
  __attribute__((enable_if((flags & (4 + 2 + 1)) == flags, "Invalid memory fence flags.")))
{
  int metal_mem_flags = __METAL_MEMORY_FLAGS_NONE__;
  if (flags & CLK_LOCAL_MEM_FENCE) {
    metal_mem_flags |= __METAL_MEMORY_FLAGS_THREADGROUP__;
  }
  if (flags & CLK_GLOBAL_MEM_FENCE) {
    metal_mem_flags |= __METAL_MEMORY_FLAGS_DEVICE__;
  }
  if (flags & CLK_IMAGE_MEM_FENCE) {
    metal_mem_flags |= __METAL_MEMORY_FLAGS_TEXTURE__;
  }
  simdgroup_barrier(metal_mem_flags);
}

#undef EXPOSE_FUNCTION_OVERLOAD_ARGS_1
#undef EXPOSE_FUNCTION_OVERLOAD_ARGS_2
#undef BRIDGE_FUNCTION_OVERLOAD_ARGS_1
#undef BRIDGE_FUNCTION_OVERLOAD_ARGS_2
#undef OVERLOAD_CLUSTERED_ARGS_1
#undef OVERLOAD_CLUSTERED_ARGS_2

#undef EXPOSE_FUNCTION_I_SCALAR_ARGS_1
#undef EXPOSE_FUNCTION_I_ARGS_1
#undef EXPOSE_FUNCTION_F_ARGS_1
#undef EXPOSE_FUNCTION_SCALAR_ARGS_2
#undef EXPOSE_FUNCTION_ARGS_2

#undef BRIDGE_FUNCTION_I_SCALAR_ARGS_1
#undef BRIDGE_FUNCTION_I_ARGS_1
#undef BRIDGE_FUNCTION_F_ARGS_1
#undef BRIDGE_FUNCTION_SCALAR_ARGS_2
#undef BRIDGE_FUNCTION_ARGS_2

#undef CLUSTERED_I_SCALAR_ARGS_1
#undef BRIDGE_FUNCTION_CLUSTERED_I_ARGS_1
#undef BRIDGE_FUNCTION_CLUSTERED_F_ARGS_1
#undef CLUSTERED_SCALAR_ARGS_2
#undef BRIDGE_FUNCTION_CLUSTERED_ARGS_2

#undef DECLARE_I_REDUCTION_BASE
#undef DECLARE_F_REDUCTION_BASE
#undef DECLARE_REDUCTION_BASE
#undef DECLARE_SHUFFLE_BASE

#undef DECLARE_I_REDUCTION_UNIFORM
#undef DECLARE_F_REDUCTION_UNIFORM
#undef DECLARE_REDUCTION_UNIFORM
#undef DECLARE_SHUFFLE_UNIFORM

#undef DECLARE_I_REDUCTION_NON_UNIFORM
#undef DECLARE_F_REDUCTION_NON_UNIFORM
#undef DECLARE_REDUCTION_NON_UNIFORM
#undef DECLARE_SHUFFLE_NON_UNIFORM

#undef DECLARE_I_REDUCTION_CLUSTERED
#undef DECLARE_F_REDUCTION_CLUSTERED
#undef DECLARE_B_REDUCTION_CLUSTERED
#undef DECLARE_REDUCTION_CLUSTERED
#undef DECLARE_SHUFFLE_CLUSTERED

#undef EXPOSE_BALLOT

#endif /* metal_stdlib_h */

#endif // VENDOR_APPLE
