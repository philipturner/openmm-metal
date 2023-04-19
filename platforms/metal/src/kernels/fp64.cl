#ifdef USE_DOUBLE_SINGLE

//
//  fp64.cl
//  OpenMM Metal Plugin
//
//  Created by Philip Turner on 3/28/23.
//

// WARNING: Compile this with '-cl-no-signed-zeroes' disabled.

// TODO: There are 703 references to 'mixed' in the common kernels. For now,
// just port the forces that I will need for my particular use case.

struct DoubleSingle {
  float hi;
  float lo;
};

struct DoubleSingle2 {
  DoubleSingle x;
  DoubleSingle y;
};

struct DoubleSingle3 {
  DoubleSingle x;
  DoubleSingle y;
  DoubleSingle z;
};

struct DoubleSingle4 {
  DoubleSingle x;
  DoubleSingle y;
  DoubleSingle z;
  DoubleSingle w;
};

typedef struct DoubleSingle DS;
typedef struct DoubleSingle2 DS2;
typedef struct DoubleSingle3 DS3;
typedef struct DoubleSingle4 DS4;

typedef struct DoubleSingle mixed;
typedef struct DoubleSingle2 mixed2;
typedef struct DoubleSingle3 mixed3;
typedef struct DoubleSingle4 mixed4;

// Initialization

DS DS_init(float hi, float lo) {
  DS output;
  output.hi = hi;
  output.lo = lo;
  return output;
}

DS3 DS3_init(DS x, DS y, DS z) {
  DS3 output;
  output.x = x;
  output.y = y;
  output.z = z;
  return output;
}

DS3 DS3_init_DS4(DS4 input) {
  DS3 output;
  output.x = input.x;
  output.y = input.y;
  output.z = input.z;
  return output;
}

DS4 DS4_init(DS x, DS y, DS z, DS w) {
  DS4 output;
  output.x = x;
  output.y = y;
  output.z = z;
  output.w = w;
  return output;
}

DS4 DS4_init_zero() {
  DS4 output;
  output.x = DS_init(0, 0);
  output.y = DS_init(0, 0);
  output.z = DS_init(0, 0);
  output.w = DS_init(0, 0);
  return output;
}

DS4 DS4_init_split(float4 hi, float4 lo) {
    DS4 output;
    output.x.hi = hi.x;
    output.y.hi = hi.y;
    output.z.hi = hi.z;
    output.w.hi = hi.w;
    
    output.x.lo = lo.x;
    output.y.lo = lo.y;
    output.z.lo = lo.z;
    output.w.lo = lo.w;
    return output;
}

DS DS_negated(DS input) {
  return DS_init(-input.hi, -input.lo);
}

DS DS_halved(DS input) {
  return DS_init(input.hi / 2, -input.lo / 2);
}

DS DS_normalized(DS input) {
  float s = input.hi + input.lo;
  float e = input.lo - (s - input.hi);
  return DS_init(s, e);
}

DS DS_init_adding(float lhs, float rhs) {
  float s = lhs + rhs;
  float v = s - lhs;
  float e = (lhs - (s - v)) + (rhs - v);
  return DS_init(s, e);
}

DS DS_init_multiplying(float lhs, float rhs) {
  float hi = lhs * rhs;
  float lo = fma(lhs, rhs, -hi);
  return DS_init(hi, lo);
}

// Addition and Subtraction

DS DS_add(DS lhs, DS rhs) {
  DS s = DS_init_adding(lhs.hi, rhs.hi);
  s.lo += lhs.lo + rhs.lo;
  return DS_normalized(s);
}

DS DS_add_float_rhs(DS lhs, float rhs) {
  DS s = DS_init_adding(lhs.hi, rhs);
  s.lo += lhs.lo;
  return DS_normalized(s);
}

DS DS_add_float_lhs(float lhs, DS rhs) {
  return DS_add_float_rhs(rhs, lhs);
}

DS DS_sub(DS lhs, DS rhs) {
  return DS_add(lhs, DS_negated(rhs));
}

DS DS_sub_float_rhs(DS lhs, float rhs) {
  return DS_add_float_rhs(lhs, -rhs);
}

DS DS_sub_float_lhs(float lhs, DS rhs) {
  return DS_add_float_lhs(lhs, DS_negated(rhs));
}

// Multiplication and Division

DS DS_mul(DS lhs, DS rhs) {
  DS p = DS_init_multiplying(lhs.hi, rhs.hi);
  p.lo = fma(lhs.hi, rhs.lo, p.lo);
  p.lo = fma(lhs.lo, rhs.hi, p.lo);
  return DS_normalized(p);
}

DS DS_mul_float_rhs(DS lhs, float rhs) {
  DS p = DS_init_multiplying(lhs.hi, rhs);
  p.lo = fma(lhs.lo, rhs, p.lo);
  return DS_normalized(p);
}

DS DS_mul_float_lhs(float lhs, DS rhs) {
  return DS_mul_float_rhs(rhs, lhs);
}

DS DS_div(DS lhs, DS rhs) {
  float xn = native_recip(rhs.hi);
  float yn = lhs.hi * xn;
  DS ayn = DS_mul_float_rhs(rhs, yn);
  float diff = DS_sub(lhs, ayn).hi;
  DS prod = DS_init_multiplying(xn, diff);
  DS q = DS_add_float_lhs(yn, prod);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  return q;
}

DS DS_recip(DS input) {
  float xn = native_recip(input.hi);
  DS ayn = DS_mul_float_rhs(input, xn);
  float diff = DS_sub_float_lhs(1, ayn).hi;
  DS prod = DS_init_multiplying(xn, diff);
  DS q = DS_add_float_lhs(xn, prod);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  return q;
}

// Transcendentals

DS DS_sqrt(DS input) {
  float xn = native_rsqrt(input.hi);
  float yn = input.hi * xn;
  DS ynsqr = DS_init_multiplying(yn, yn);
  float diff = DS_sub(input, ynsqr).hi;
  DS prod = DS_init_multiplying(xn, diff);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  DS output = DS_sub_float_lhs(yn, DS_halved(prod));
  return (input.hi == 0) ? DS_init(0, 0) : output;
}

DS DS_rsqrt(DS input) {
  float xn = native_rsqrt(input.hi);
  DS xn2 = DS_init_multiplying(xn, xn);
  DS y2n = DS_mul(input, xn2);
  float diff = DS_sub_float_lhs(1, y2n).hi;
  DS prod = DS_init_multiplying(xn, diff);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  return DS_sub_float_lhs(xn, DS_halved(prod));
}

DS DS_rsqrt_special(DS input) {
  float xn = native_rsqrt(input.hi);
  DS xn2 = DS_init_multiplying(xn, xn);
  DS y2n = DS_mul(input, xn2);
  float diff = DS_sub_float_lhs(1, y2n).hi;
  DS prod = DS_init_multiplying(xn, diff);
  
  // Don't handle infinity case because any `INF` will cause undefined
  // behavior in other code anyway.
  DS output = DS_sub_float_lhs(xn, DS_halved(prod));
  return (input.hi == 0) ? DS_init(0, 0) : output;
}

// Conversions

DS DS_init_long_slow(long value) {
  DS output;
  output.hi = float(value);
  
  // If you can guarantee this never happens, it will improve performance.
  if (fabs(output.hi) >= float(__LONG_MAX__)) {
    output.lo = float(0);
  } else {
    output.lo = float(value - long(output.hi));
  }
  return output;
}

DS DS_init_long_fast(long value) {
  DS output;
  output.hi = float(value);
  output.lo = float(value - long(output.hi));
  return output;
}

DS DS_init_ulong_slow(ulong value) {
  DS output;
  output.hi = float(value);
  
  // If you can guarantee this never happens, it will improve performance.
  if (fabs(output.hi) >= float(__LONG_MAX__ * 2)) {
    output.lo = float(0);
  } else {
    output.lo = float(value - ulong(output.hi));
  }
  return output;
}

DS DS_init_ulong_fast(long value) {
  DS output;
  output.hi = float(value);
  output.lo = float(value - ulong(output.hi));
  return output;
}

// Never assume positions are correctly normalized.
#define FP64_APPLY_POS_DELTA \
real4 pos1 = posq[index]; \
real4 pos2 = posqCorrection[index]; \
mixed4 pos = DS4_init(DS_init_adding(pos1.x, pos2.x), \
                      DS_init_adding(pos1.y, pos2.y), \
                      DS_init_adding(pos1.z, pos2.z), \
                      DS_init(pos1.w, 0)); \
\
mixed3 delta = *((GLOBAL mixed3*)(posDelta + index)); \
pos.x = DS_add(pos.x, delta.x); \
pos.y = DS_add(pos.y, delta.y); \
pos.z = DS_add(pos.y, delta.z); \
posq[index] = make_real4(pos.x.hi, pos.y.hi, pos.z.hi, pos.w.hi); \
posqCorrection[index] = make_real4(pos.x.lo, pos.y.lo, pos.z.lo, 0); \

#endif // USE_DOUBLE_SINGLE