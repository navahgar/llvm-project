/*===---- avx2intrin.h - AVX2 intrinsics -----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <avx2intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX2INTRIN_H
#define __AVX2INTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS256 __attribute__((__always_inline__, __nodebug__, __target__("avx2"), __min_vector_width__(256)))
#define __DEFAULT_FN_ATTRS128 __attribute__((__always_inline__, __nodebug__, __target__("avx2"), __min_vector_width__(128)))

/* SSE4 Multiple Packed Sums of Absolute Difference.  */
#define _mm256_mpsadbw_epu8(X, Y, M) \
  ((__m256i)__builtin_ia32_mpsadbw256((__v32qi)(__m256i)(X), \
                                      (__v32qi)(__m256i)(Y), (int)(M)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_abs_epi8(__m256i __a)
{
    return (__m256i)__builtin_elementwise_abs((__v32qs)__a);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_abs_epi16(__m256i __a)
{
    return (__m256i)__builtin_elementwise_abs((__v16hi)__a);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_abs_epi32(__m256i __a)
{
    return (__m256i)__builtin_elementwise_abs((__v8si)__a);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_packs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packsswb256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_packs_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packssdw256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_packus_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packuswb256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_packus_epi32(__m256i __V1, __m256i __V2)
{
  return (__m256i) __builtin_ia32_packusdw256((__v8si)__V1, (__v8si)__V2);
}

/// Adds 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors and returns the lower 8 bits of each sum in the corresponding
///    byte of the 256-bit integer vector result (overflow is ignored).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDB instruction.
///
/// \param __a
///    A 256-bit integer vector containing one of the source operands.
/// \param __b
///    A 256-bit integer vector containing one of the source operands.
/// \returns A 256-bit integer vector containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_add_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qu)__a + (__v32qu)__b);
}

/// Adds 16-bit integers from corresponding elements of two 256-bit vectors of
///    [16 x i16] and returns the lower 16 bits of each sum in the
///    corresponding element of the [16 x i16] result (overflow is ignored).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_add_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hu)__a + (__v16hu)__b);
}

/// Adds 32-bit integers from corresponding elements of two 256-bit vectors of
///    [8 x i32] and returns the lower 32 bits of each sum in the corresponding
///    element of the [8 x i32] result (overflow is ignored).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [8 x i32] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_add_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8su)__a + (__v8su)__b);
}

/// Adds 64-bit integers from corresponding elements of two 256-bit vectors of
///    [4 x i64] and returns the lower 64 bits of each sum in the corresponding
///    element of the [4 x i64] result (overflow is ignored).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [4 x i64] containing one of the source operands.
/// \returns A 256-bit vector of [4 x i64] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_add_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a + (__v4du)__b);
}

/// Adds 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors using signed saturation, and returns each sum in the
///    corresponding byte of the 256-bit integer vector result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDSB instruction.
///
/// \param __a
///    A 256-bit integer vector containing one of the source operands.
/// \param __b
///    A 256-bit integer vector containing one of the source operands.
/// \returns A 256-bit integer vector containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_adds_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_add_sat((__v32qs)__a, (__v32qs)__b);
}

/// Adds 16-bit integers from corresponding elements of two 256-bit vectors of
///    [16 x i16] using signed saturation, and returns the [16 x i16] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_adds_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_add_sat((__v16hi)__a, (__v16hi)__b);
}

/// Adds 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors using unsigned saturation, and returns each sum in the
///    corresponding byte of the 256-bit integer vector result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDUSB instruction.
///
/// \param __a
///    A 256-bit integer vector containing one of the source operands.
/// \param __b
///    A 256-bit integer vector containing one of the source operands.
/// \returns A 256-bit integer vector containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_adds_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_add_sat((__v32qu)__a, (__v32qu)__b);
}

/// Adds 16-bit integers from corresponding elements of two 256-bit vectors of
///    [16 x i16] using unsigned saturation, and returns the [16 x i16] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPADDUSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_adds_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_add_sat((__v16hu)__a, (__v16hu)__b);
}

#define _mm256_alignr_epi8(a, b, n) \
  ((__m256i)__builtin_ia32_palignr256((__v32qi)(__m256i)(a), \
                                      (__v32qi)(__m256i)(b), (n)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_and_si256(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a & (__v4du)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_andnot_si256(__m256i __a, __m256i __b)
{
  return (__m256i)(~(__v4du)__a & (__v4du)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_avg_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pavgb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_avg_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pavgw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_blendv_epi8(__m256i __V1, __m256i __V2, __m256i __M)
{
  return (__m256i)__builtin_ia32_pblendvb256((__v32qi)__V1, (__v32qi)__V2,
                                              (__v32qi)__M);
}

#define _mm256_blend_epi16(V1, V2, M) \
  ((__m256i)__builtin_ia32_pblendw256((__v16hi)(__m256i)(V1), \
                                      (__v16hi)(__m256i)(V2), (int)(M)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a == (__v32qi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a == (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a == (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4di)__a == (__v4di)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpgt_epi8(__m256i __a, __m256i __b)
{
  /* This function always performs a signed comparison, but __v32qi is a char
     which may be signed or unsigned, so use __v32qs. */
  return (__m256i)((__v32qs)__a > (__v32qs)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpgt_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a > (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpgt_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a > (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpgt_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4di)__a > (__v4di)__b);
}

/// Horizontally adds the adjacent pairs of 16-bit integers from two 256-bit
///    vectors of [16 x i16] and returns the lower 16 bits of each sum in an
///    element of the [16 x i16] result (overflow is ignored). Sums from
///    \a __a are returned in the lower 64 bits of each 128-bit half of the
///    result; sums from \a __b are returned in the upper 64 bits of each
///    128-bit half of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+15:j] := __a[j+15:j] + __a[j+31:j+16]
///   result[j+31:j+16] := __a[j+47:j+32] + __a[j+63:j+48]
///   result[j+47:j+32] := __a[j+79:j+64] + __a[j+95:j+80]
///   result[j+63:j+48] := __a[j+111:j+96] + __a[j+127:j+112]
///   result[j+79:j+64] := __b[j+15:j] + __b[j+31:j+16]
///   result[j+95:j+80] := __b[j+47:j+32] + __b[j+63:j+48]
///   result[j+111:j+96] := __b[j+79:j+64] + __b[j+95:j+80]
///   result[j+127:j+112] := __b[j+111:j+96] + __b[j+127:j+112]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHADDW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hadd_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddw256((__v16hi)__a, (__v16hi)__b);
}

/// Horizontally adds the adjacent pairs of 32-bit integers from two 256-bit
///    vectors of [8 x i32] and returns the lower 32 bits of each sum in an
///    element of the [8 x i32] result (overflow is ignored). Sums from \a __a
///    are returned in the lower 64 bits of each 128-bit half of the result;
///    sums from \a __b are returned in the upper 64 bits of each 128-bit half
///    of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+31:j] := __a[j+31:j] + __a[j+63:j+32]
///   result[j+63:j+32] := __a[j+95:j+64] + __a[j+127:j+96]
///   result[j+95:j+64] := __b[j+31:j] + __b[j+63:j+32]
///   result[j+127:j+96] := __b[j+95:j+64] + __b[j+127:j+96]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHADDD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [8 x i32] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hadd_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddd256((__v8si)__a, (__v8si)__b);
}

/// Horizontally adds the adjacent pairs of 16-bit integers from two 256-bit
///    vectors of [16 x i16] using signed saturation and returns each sum in
///    an element of the [16 x i16] result. Sums from \a __a are returned in
///    the lower 64 bits of each 128-bit half of the result; sums from \a __b
///    are returned in the upper 64 bits of each 128-bit half of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+15:j] := SATURATE16(__a[j+15:j] + __a[j+31:j+16])
///   result[j+31:j+16] := SATURATE16(__a[j+47:j+32] + __a[j+63:j+48])
///   result[j+47:j+32] := SATURATE16(__a[j+79:j+64] + __a[j+95:j+80])
///   result[j+63:j+48] := SATURATE16(__a[j+111:j+96] + __a[j+127:j+112])
///   result[j+79:j+64] := SATURATE16(__b[j+15:j] + __b[j+31:j+16])
///   result[j+95:j+80] := SATURATE16(__b[j+47:j+32] + __b[j+63:j+48])
///   result[j+111:j+96] := SATURATE16(__b[j+79:j+64] + __b[j+95:j+80])
///   result[j+127:j+112] := SATURATE16(__b[j+111:j+96] + __b[j+127:j+112])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHADDSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the sums.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hadds_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddsw256((__v16hi)__a, (__v16hi)__b);
}

/// Horizontally subtracts adjacent pairs of 16-bit integers from two 256-bit
///    vectors of [16 x i16] and returns the lower 16 bits of each difference
///    in an element of the [16 x i16] result (overflow is ignored).
///    Differences from \a __a are returned in the lower 64 bits of each
///    128-bit half of the result; differences from \a __b are returned in the
///    upper 64 bits of each 128-bit half of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+15:j] := __a[j+15:j] - __a[j+31:j+16]
///   result[j+31:j+16] := __a[j+47:j+32] - __a[j+63:j+48]
///   result[j+47:j+32] := __a[j+79:j+64] - __a[j+95:j+80]
///   result[j+63:j+48] := __a[j+111:j+96] - __a[j+127:j+112]
///   result[j+79:j+64] := __b[j+15:j] - __b[j+31:j+16]
///   result[j+95:j+80] := __b[j+47:j+32] - __b[j+63:j+48]
///   result[j+111:j+96] := __b[j+79:j+64] - __b[j+95:j+80]
///   result[j+127:j+112] := __b[j+111:j+96] - __b[j+127:j+112]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHSUBW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hsub_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubw256((__v16hi)__a, (__v16hi)__b);
}

/// Horizontally subtracts adjacent pairs of 32-bit integers from two 256-bit
///    vectors of [8 x i32] and returns the lower 32 bits of each difference in
///    an element of the [8 x i32] result (overflow is ignored). Differences
///    from \a __a are returned in the lower 64 bits of each 128-bit half of
///    the result; differences from \a __b are returned in the upper 64 bits
///    of each 128-bit half of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+31:j] := __a[j+31:j] - __a[j+63:j+32]
///   result[j+63:j+32] := __a[j+95:j+64] - __a[j+127:j+96]
///   result[j+95:j+64] := __b[j+31:j] - __b[j+63:j+32]
///   result[j+127:j+96] := __b[j+95:j+64] - __b[j+127:j+96]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHSUBD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [8 x i32] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hsub_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubd256((__v8si)__a, (__v8si)__b);
}

/// Horizontally subtracts adjacent pairs of 16-bit integers from two 256-bit
///    vectors of [16 x i16] using signed saturation and returns each sum in
///    an element of the [16 x i16] result. Differences from \a __a are
///    returned in the lower 64 bits of each 128-bit half of the result;
///    differences from \a __b are returned in the upper 64 bits of each
///    128-bit half of the result.
///
/// \code{.operation}
/// FOR i := 0 TO 1
///   j := i*128
///   result[j+15:j] := SATURATE16(__a[j+15:j] - __a[j+31:j+16])
///   result[j+31:j+16] := SATURATE16(__a[j+47:j+32] - __a[j+63:j+48])
///   result[j+47:j+32] := SATURATE16(__a[j+79:j+64] - __a[j+95:j+80])
///   result[j+63:j+48] := SATURATE16(__a[j+111:j+96] - __a[j+127:j+112])
///   result[j+79:j+64] := SATURATE16(__b[j+15:j] - __b[j+31:j+16])
///   result[j+95:j+80] := SATURATE16(__b[j+47:j+32] - __b[j+63:j+48])
///   result[j+111:j+96] := SATURATE16(__b[j+79:j+64] - __b[j+95:j+80])
///   result[j+127:j+112] := SATURATE16(__b[j+111:j+96] - __b[j+127:j+112])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPHSUBSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_hsubs_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubsw256((__v16hi)__a, (__v16hi)__b);
}

/// Multiplies each unsigned byte from the 256-bit integer vector in \a __a
///    with the corresponding signed byte from the 256-bit integer vector in
///    \a __b, forming signed 16-bit intermediate products. Adds adjacent
///    pairs of those products using signed saturation to form 16-bit sums
///    returned as elements of the [16 x i16] result.
///
/// \code{.operation}
/// FOR i := 0 TO 15
///   j := i*16
///   temp1 := __a[j+7:j] * __b[j+7:j]
///   temp2 := __a[j+15:j+8] * __b[j+15:j+8]
///   result[j+15:j] := SATURATE16(temp1 + temp2)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMADDUBSW instruction.
///
/// \param __a
///    A 256-bit vector containing one of the source operands.
/// \param __b
///    A 256-bit vector containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maddubs_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_pmaddubsw256((__v32qi)__a, (__v32qi)__b);
}

/// Multiplies corresponding 16-bit elements of two 256-bit vectors of
///    [16 x i16], forming 32-bit intermediate products, and adds pairs of
///    those products to form 32-bit sums returned as elements of the
///    [8 x i32] result.
///
///    There is only one wraparound case: when all four of the 16-bit sources
///    are \c 0x8000, the result will be \c 0x80000000.
///
/// \code{.operation}
/// FOR i := 0 TO 7
///   j := i*32
///   temp1 := __a[j+15:j] * __b[j+15:j]
///   temp2 := __a[j+31:j+16] * __b[j+31:j+16]
///   result[j+31:j] := temp1 + temp2
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMADDWD instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_madd_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaddwd256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v32qs)__a, (__v32qs)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v32qu)__a, (__v32qu)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v16hu)__a, (__v16hu)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_max_epu32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_max((__v8su)__a, (__v8su)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v32qs)__a, (__v32qs)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v32qu)__a, (__v32qu)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v16hu)__a, (__v16hu)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_min_epu32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_min((__v8su)__a, (__v8su)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS256
_mm256_movemask_epi8(__m256i __a)
{
  return __builtin_ia32_pmovmskb256((__v32qi)__a);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi8_epi16(__m128i __V)
{
  /* This function always performs a signed extension, but __v16qi is a char
     which may be signed or unsigned, so use __v16qs. */
  return (__m256i)__builtin_convertvector((__v16qs)__V, __v16hi);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi8_epi32(__m128i __V)
{
  /* This function always performs a signed extension, but __v16qi is a char
     which may be signed or unsigned, so use __v16qs. */
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8si);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi8_epi64(__m128i __V)
{
  /* This function always performs a signed extension, but __v16qi is a char
     which may be signed or unsigned, so use __v16qs. */
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v16qs)__V, (__v16qs)__V, 0, 1, 2, 3), __v4di);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi16_epi32(__m128i __V)
{
  return (__m256i)__builtin_convertvector((__v8hi)__V, __v8si);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi16_epi64(__m128i __V)
{
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v8hi)__V, (__v8hi)__V, 0, 1, 2, 3), __v4di);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepi32_epi64(__m128i __V)
{
  return (__m256i)__builtin_convertvector((__v4si)__V, __v4di);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu8_epi16(__m128i __V)
{
  return (__m256i)__builtin_convertvector((__v16qu)__V, __v16hi);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu8_epi32(__m128i __V)
{
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3, 4, 5, 6, 7), __v8si);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu8_epi64(__m128i __V)
{
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v16qu)__V, (__v16qu)__V, 0, 1, 2, 3), __v4di);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu16_epi32(__m128i __V)
{
  return (__m256i)__builtin_convertvector((__v8hu)__V, __v8si);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu16_epi64(__m128i __V)
{
  return (__m256i)__builtin_convertvector(__builtin_shufflevector((__v8hu)__V, (__v8hu)__V, 0, 1, 2, 3), __v4di);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtepu32_epi64(__m128i __V)
{
  return (__m256i)__builtin_convertvector((__v4su)__V, __v4di);
}

/// Multiplies signed 32-bit integers from even-numbered elements of two
///    256-bit vectors of [8 x i32] and returns the 64-bit products in the
///    [4 x i64] result.
///
/// \code{.operation}
/// result[63:0] := __a[31:0] * __b[31:0]
/// result[127:64] := __a[95:64] * __b[95:64]
/// result[191:128] := __a[159:128] * __b[159:128]
/// result[255:192] := __a[223:192] * __b[223:192]
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULDQ instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [4 x i64] containing the products.
static __inline__  __m256i __DEFAULT_FN_ATTRS256
_mm256_mul_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmuldq256((__v8si)__a, (__v8si)__b);
}

/// Multiplies signed 16-bit integer elements of two 256-bit vectors of
///    [16 x i16], truncates the 32-bit results to the most significant 18
///    bits, rounds by adding 1, and returns bits [16:1] of each rounded
///    product in the [16 x i16] result.
///
/// \code{.operation}
/// FOR i := 0 TO 15
///   j := i*16
///   temp := ((__a[j+15:j] * __b[j+15:j]) >> 14) + 1
///   result[j+15:j] := temp[16:1]
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULHRSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the rounded products.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mulhrs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhrsw256((__v16hi)__a, (__v16hi)__b);
}

/// Multiplies unsigned 16-bit integer elements of two 256-bit vectors of
///    [16 x i16], and returns the upper 16 bits of each 32-bit product in the
///    [16 x i16] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULHUW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the products.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mulhi_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhuw256((__v16hi)__a, (__v16hi)__b);
}

/// Multiplies signed 16-bit integer elements of two 256-bit vectors of
///    [16 x i16], and returns the upper 16 bits of each 32-bit product in the
///    [16 x i16] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULHW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the products.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mulhi_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhw256((__v16hi)__a, (__v16hi)__b);
}

/// Multiplies signed 16-bit integer elements of two 256-bit vectors of
///    [16 x i16], and returns the lower 16 bits of each 32-bit product in the
///    [16 x i16] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULLW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [16 x i16] containing one of the source operands.
/// \returns A 256-bit vector of [16 x i16] containing the products.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mullo_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hu)__a * (__v16hu)__b);
}

/// Multiplies signed 32-bit integer elements of two 256-bit vectors of
///    [8 x i32], and returns the lower 32 bits of each 64-bit product in the
///    [8 x i32] result.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULLD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [8 x i32] containing the products.
static __inline__  __m256i __DEFAULT_FN_ATTRS256
_mm256_mullo_epi32 (__m256i __a, __m256i __b)
{
  return (__m256i)((__v8su)__a * (__v8su)__b);
}

/// Multiplies unsigned 32-bit integers from even-numered elements of two
///    256-bit vectors of [8 x i32] and returns the 64-bit products in the
///    [4 x i64] result.
///
/// \code{.operation}
/// result[63:0] := __a[31:0] * __b[31:0]
/// result[127:64] := __a[95:64] * __b[95:64]
/// result[191:128] := __a[159:128] * __b[159:128]
/// result[255:192] := __a[223:192] * __b[223:192]
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMULUDQ instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \param __b
///    A 256-bit vector of [8 x i32] containing one of the source operands.
/// \returns A 256-bit vector of [4 x i64] containing the products.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mul_epu32(__m256i __a, __m256i __b)
{
  return __builtin_ia32_pmuludq256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_or_si256(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a | (__v4du)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sad_epu8(__m256i __a, __m256i __b)
{
  return __builtin_ia32_psadbw256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_shuffle_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pshufb256((__v32qi)__a, (__v32qi)__b);
}

#define _mm256_shuffle_epi32(a, imm) \
  ((__m256i)__builtin_ia32_pshufd256((__v8si)(__m256i)(a), (int)(imm)))

#define _mm256_shufflehi_epi16(a, imm) \
  ((__m256i)__builtin_ia32_pshufhw256((__v16hi)(__m256i)(a), (int)(imm)))

#define _mm256_shufflelo_epi16(a, imm) \
  ((__m256i)__builtin_ia32_pshuflw256((__v16hi)(__m256i)(a), (int)(imm)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sign_epi8(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sign_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sign_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignd256((__v8si)__a, (__v8si)__b);
}

/// Shifts each 128-bit half of the 256-bit integer vector \a a left by
///    \a imm bytes, shifting in zero bytes, and returns the result. If \a imm
///    is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_slli_si256(__m256i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSLLDQ instruction.
///
/// \param a
///    A 256-bit integer vector to be shifted.
/// \param imm
///     An unsigned immediate value specifying the shift count (in bytes).
/// \returns A 256-bit integer vector containing the result.
#define _mm256_slli_si256(a, imm) \
  ((__m256i)__builtin_ia32_pslldqi256_byteshift((__v4di)(__m256i)(a), (int)(imm)))

/// Shifts each 128-bit half of the 256-bit integer vector \a a left by
///    \a imm bytes, shifting in zero bytes, and returns the result. If \a imm
///    is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_bslli_epi128(__m256i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSLLDQ instruction.
///
/// \param a
///    A 256-bit integer vector to be shifted.
/// \param imm
///    An unsigned immediate value specifying the shift count (in bytes).
/// \returns A 256-bit integer vector containing the result.
#define _mm256_bslli_epi128(a, imm) \
  ((__m256i)__builtin_ia32_pslldqi256_byteshift((__v4di)(__m256i)(a), (int)(imm)))

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    left by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_slli_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psllwi256((__v16hi)__a, __count);
}

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    left by the number of bits specified by the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sll_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psllw256((__v16hi)__a, (__v8hi)__count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    left by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 31, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_slli_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_pslldi256((__v8si)__a, __count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    left by the number of bits given in the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 31, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sll_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_pslld256((__v8si)__a, (__v4si)__count);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __a
///    left by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 63, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_slli_epi64(__m256i __a, int __count)
{
  return __builtin_ia32_psllqi256((__v4di)__a, __count);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __a
///    left by the number of bits given in the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 63, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sll_epi64(__m256i __a, __m128i __count)
{
  return __builtin_ia32_psllq256((__v4di)__a, __count);
}

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    right by \a __count bits, shifting in sign bits, and returns the result.
///    If \a __count is greater than 15, each element of the result is either
///    0 or -1 according to the corresponding input sign bit.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srai_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrawi256((__v16hi)__a, __count);
}

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    right by the number of bits given in the lower 64 bits of \a __count,
///    shifting in sign bits, and returns the result. If \a __count is greater
///    than 15, each element of the result is either 0 or -1 according to the
///    corresponding input sign bit.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sra_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psraw256((__v16hi)__a, (__v8hi)__count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    right by \a __count bits, shifting in sign bits, and returns the result.
///    If \a __count is greater than 31, each element of the result is either
///    0 or -1 according to the corresponding input sign bit.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srai_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psradi256((__v8si)__a, __count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    right by the number of bits given in the lower 64 bits of \a __count,
///    shifting in sign bits, and returns the result. If \a __count is greater
///    than 31, each element of the result is either 0 or -1 according to the
///    corresponding input sign bit.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sra_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrad256((__v8si)__a, (__v4si)__count);
}

/// Shifts each 128-bit half of the 256-bit integer vector in \a a right by
///    \a imm bytes, shifting in zero bytes, and returns the result. If
///    \a imm is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_srli_si256(__m256i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSRLDQ instruction.
///
/// \param a
///    A 256-bit integer vector to be shifted.
/// \param imm
///    An unsigned immediate value specifying the shift count (in bytes).
/// \returns A 256-bit integer vector containing the result.
#define _mm256_srli_si256(a, imm) \
  ((__m256i)__builtin_ia32_psrldqi256_byteshift((__m256i)(a), (int)(imm)))

/// Shifts each 128-bit half of the 256-bit integer vector in \a a right by
///    \a imm bytes, shifting in zero bytes, and returns the result. If
///    \a imm is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_bsrli_epi128(__m256i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSRLDQ instruction.
///
/// \param a
///    A 256-bit integer vector to be shifted.
/// \param imm
///     An unsigned immediate value specifying the shift count (in bytes).
/// \returns A 256-bit integer vector containing the result.
#define _mm256_bsrli_epi128(a, imm) \
  ((__m256i)__builtin_ia32_psrldqi256_byteshift((__m256i)(a), (int)(imm)))

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    right by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srli_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrlwi256((__v16hi)__a, __count);
}

/// Shifts each 16-bit element of the 256-bit vector of [16 x i16] in \a __a
///    right by the number of bits given in the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 15, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [16 x i16] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srl_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrlw256((__v16hi)__a, (__v8hi)__count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    right by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 31, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srli_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrldi256((__v8si)__a, __count);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __a
///    right by the number of bits given in the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 31, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srl_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrld256((__v8si)__a, (__v4si)__count);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __a
///    right by \a __count bits, shifting in zero bits, and returns the result.
///    If \a __count is greater than 63, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __count
///    An unsigned integer value specifying the shift count (in bits).
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srli_epi64(__m256i __a, int __count)
{
  return __builtin_ia32_psrlqi256((__v4di)__a, __count);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __a
///    right by the number of bits given in the lower 64 bits of \a __count,
///    shifting in zero bits, and returns the result. If \a __count is greater
///    than 63, the returned result is all zeroes.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __count
///    A 128-bit vector of [2 x i64] whose lower element gives the unsigned
///    shift count (in bits). The upper element is ignored.
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srl_epi64(__m256i __a, __m128i __count)
{
  return __builtin_ia32_psrlq256((__v4di)__a, __count);
}

/// Subtracts 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors. Returns the lower 8 bits of each difference in the
///    corresponding byte of the 256-bit integer vector result (overflow is
///    ignored).
///
/// \code{.operation}
/// FOR i := 0 TO 31
///   j := i*8
///   result[j+7:j] := __a[j+7:j] - __b[j+7:j]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBB instruction.
///
/// \param __a
///    A 256-bit integer vector containing the minuends.
/// \param __b
///    A 256-bit integer vector containing the subtrahends.
/// \returns A 256-bit integer vector containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sub_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qu)__a - (__v32qu)__b);
}

/// Subtracts 16-bit integers from corresponding elements of two 256-bit
///    vectors of [16 x i16]. Returns the lower 16 bits of each difference in
///    the corresponding element of the [16 x i16] result (overflow is
///    ignored).
///
/// \code{.operation}
/// FOR i := 0 TO 15
///   j := i*16
///   result[j+15:j] := __a[j+15:j] - __b[j+15:j]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing the minuends.
/// \param __b
///    A 256-bit vector of [16 x i16] containing the subtrahends.
/// \returns A 256-bit vector of [16 x i16] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sub_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hu)__a - (__v16hu)__b);
}

/// Subtracts 32-bit integers from corresponding elements of two 256-bit
///    vectors of [8 x i32]. Returns the lower 32 bits of each difference in
///    the corresponding element of the [8 x i32] result (overflow is ignored).
///
/// \code{.operation}
/// FOR i := 0 TO 7
///   j := i*32
///   result[j+31:j] := __a[j+31:j] - __b[j+31:j]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBD instruction.
///
/// \param __a
///    A 256-bit vector of [8 x i32] containing the minuends.
/// \param __b
///    A 256-bit vector of [8 x i32] containing the subtrahends.
/// \returns A 256-bit vector of [8 x i32] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sub_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8su)__a - (__v8su)__b);
}

/// Subtracts 64-bit integers from corresponding elements of two 256-bit
///    vectors of [4 x i64]. Returns the lower 64 bits of each difference in
///    the corresponding element of the [4 x i64] result (overflow is ignored).
///
/// \code{.operation}
/// FOR i := 0 TO 3
///   j := i*64
///   result[j+63:j] := __a[j+63:j] - __b[j+63:j]
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBQ instruction.
///
/// \param __a
///    A 256-bit vector of [4 x i64] containing the minuends.
/// \param __b
///    A 256-bit vector of [4 x i64] containing the subtrahends.
/// \returns A 256-bit vector of [4 x i64] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sub_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a - (__v4du)__b);
}

/// Subtracts 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors using signed saturation, and returns each differences in the
///    corresponding byte of the 256-bit integer vector result.
///
/// \code{.operation}
/// FOR i := 0 TO 31
///   j := i*8
///   result[j+7:j] := SATURATE8(__a[j+7:j] - __b[j+7:j])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBSB instruction.
///
/// \param __a
///    A 256-bit integer vector containing the minuends.
/// \param __b
///    A 256-bit integer vector containing the subtrahends.
/// \returns A 256-bit integer vector containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_subs_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_sub_sat((__v32qs)__a, (__v32qs)__b);
}

/// Subtracts 16-bit integers from corresponding elements of two 256-bit
///    vectors of [16 x i16] using signed saturation, and returns each
///    difference in the corresponding element of the [16 x i16] result.
///
/// \code{.operation}
/// FOR i := 0 TO 15
///   j := i*16
///   result[j+7:j] := SATURATE16(__a[j+7:j] - __b[j+7:j])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing the minuends.
/// \param __b
///    A 256-bit vector of [16 x i16] containing the subtrahends.
/// \returns A 256-bit vector of [16 x i16] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_subs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_sub_sat((__v16hi)__a, (__v16hi)__b);
}

/// Subtracts 8-bit integers from corresponding bytes of two 256-bit integer
///    vectors using unsigned saturation, and returns each difference in the
///    corresponding byte of the 256-bit integer vector result. For each byte,
///    computes <c> result = __a - __b </c>.
///
/// \code{.operation}
/// FOR i := 0 TO 31
///   j := i*8
///   result[j+7:j] := SATURATE8U(__a[j+7:j] - __b[j+7:j])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBUSB instruction.
///
/// \param __a
///    A 256-bit integer vector containing the minuends.
/// \param __b
///    A 256-bit integer vector containing the subtrahends.
/// \returns A 256-bit integer vector containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_subs_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_sub_sat((__v32qu)__a, (__v32qu)__b);
}

/// Subtracts 16-bit integers from corresponding elements of two 256-bit
///    vectors of [16 x i16] using unsigned saturation, and returns each
///    difference in the corresponding element of the [16 x i16] result.
///
/// \code{.operation}
/// FOR i := 0 TO 15
///   j := i*16
///   result[j+15:j] := SATURATE16U(__a[j+15:j] - __b[j+15:j])
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSUBUSW instruction.
///
/// \param __a
///    A 256-bit vector of [16 x i16] containing the minuends.
/// \param __b
///    A 256-bit vector of [16 x i16] containing the subtrahends.
/// \returns A 256-bit vector of [16 x i16] containing the differences.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_subs_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_elementwise_sub_sat((__v16hu)__a, (__v16hu)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpackhi_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v32qi)__a, (__v32qi)__b, 8, 32+8, 9, 32+9, 10, 32+10, 11, 32+11, 12, 32+12, 13, 32+13, 14, 32+14, 15, 32+15, 24, 32+24, 25, 32+25, 26, 32+26, 27, 32+27, 28, 32+28, 29, 32+29, 30, 32+30, 31, 32+31);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpackhi_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)__b, 4, 16+4, 5, 16+5, 6, 16+6, 7, 16+7, 12, 16+12, 13, 16+13, 14, 16+14, 15, 16+15);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpackhi_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v8si)__a, (__v8si)__b, 2, 8+2, 3, 8+3, 6, 8+6, 7, 8+7);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpackhi_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v4di)__a, (__v4di)__b, 1, 4+1, 3, 4+3);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpacklo_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v32qi)__a, (__v32qi)__b, 0, 32+0, 1, 32+1, 2, 32+2, 3, 32+3, 4, 32+4, 5, 32+5, 6, 32+6, 7, 32+7, 16, 32+16, 17, 32+17, 18, 32+18, 19, 32+19, 20, 32+20, 21, 32+21, 22, 32+22, 23, 32+23);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpacklo_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)__b, 0, 16+0, 1, 16+1, 2, 16+2, 3, 16+3, 8, 16+8, 9, 16+9, 10, 16+10, 11, 16+11);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpacklo_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v8si)__a, (__v8si)__b, 0, 8+0, 1, 8+1, 4, 8+4, 5, 8+5);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_unpacklo_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v4di)__a, (__v4di)__b, 0, 4+0, 2, 4+2);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_xor_si256(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a ^ (__v4du)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_stream_load_si256(__m256i const *__V)
{
  typedef __v4di __v4di_aligned __attribute__((aligned(32)));
  return (__m256i)__builtin_nontemporal_load((const __v4di_aligned *)__V);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_broadcastss_ps(__m128 __X)
{
  return (__m128)__builtin_shufflevector((__v4sf)__X, (__v4sf)__X, 0, 0, 0, 0);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS128
_mm_broadcastsd_pd(__m128d __a)
{
  return __builtin_shufflevector((__v2df)__a, (__v2df)__a, 0, 0);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_broadcastss_ps(__m128 __X)
{
  return (__m256)__builtin_shufflevector((__v4sf)__X, (__v4sf)__X, 0, 0, 0, 0, 0, 0, 0, 0);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS256
_mm256_broadcastsd_pd(__m128d __X)
{
  return (__m256d)__builtin_shufflevector((__v2df)__X, (__v2df)__X, 0, 0, 0, 0);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_broadcastsi128_si256(__m128i __X)
{
  return (__m256i)__builtin_shufflevector((__v2di)__X, (__v2di)__X, 0, 1, 0, 1);
}

#define _mm_broadcastsi128_si256(X) _mm256_broadcastsi128_si256(X)

#define _mm_blend_epi32(V1, V2, M) \
  ((__m128i)__builtin_ia32_pblendd128((__v4si)(__m128i)(V1), \
                                      (__v4si)(__m128i)(V2), (int)(M)))

#define _mm256_blend_epi32(V1, V2, M) \
  ((__m256i)__builtin_ia32_pblendd256((__v8si)(__m256i)(V1), \
                                      (__v8si)(__m256i)(V2), (int)(M)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_broadcastb_epi8(__m128i __X)
{
  return (__m256i)__builtin_shufflevector((__v16qi)__X, (__v16qi)__X, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_broadcastw_epi16(__m128i __X)
{
  return (__m256i)__builtin_shufflevector((__v8hi)__X, (__v8hi)__X, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_broadcastd_epi32(__m128i __X)
{
  return (__m256i)__builtin_shufflevector((__v4si)__X, (__v4si)__X, 0, 0, 0, 0, 0, 0, 0, 0);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_broadcastq_epi64(__m128i __X)
{
  return (__m256i)__builtin_shufflevector((__v2di)__X, (__v2di)__X, 0, 0, 0, 0);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_broadcastb_epi8(__m128i __X)
{
  return (__m128i)__builtin_shufflevector((__v16qi)__X, (__v16qi)__X, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_broadcastw_epi16(__m128i __X)
{
  return (__m128i)__builtin_shufflevector((__v8hi)__X, (__v8hi)__X, 0, 0, 0, 0, 0, 0, 0, 0);
}


static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_broadcastd_epi32(__m128i __X)
{
  return (__m128i)__builtin_shufflevector((__v4si)__X, (__v4si)__X, 0, 0, 0, 0);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_broadcastq_epi64(__m128i __X)
{
  return (__m128i)__builtin_shufflevector((__v2di)__X, (__v2di)__X, 0, 0);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_permutevar8x32_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_permvarsi256((__v8si)__a, (__v8si)__b);
}

#define _mm256_permute4x64_pd(V, M) \
  ((__m256d)__builtin_ia32_permdf256((__v4df)(__m256d)(V), (int)(M)))

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_permutevar8x32_ps(__m256 __a, __m256i __b)
{
  return (__m256)__builtin_ia32_permvarsf256((__v8sf)__a, (__v8si)__b);
}

#define _mm256_permute4x64_epi64(V, M) \
  ((__m256i)__builtin_ia32_permdi256((__v4di)(__m256i)(V), (int)(M)))

#define _mm256_permute2x128_si256(V1, V2, M) \
  ((__m256i)__builtin_ia32_permti256((__m256i)(V1), (__m256i)(V2), (int)(M)))

#define _mm256_extracti128_si256(V, M) \
  ((__m128i)__builtin_ia32_extract128i256((__v4di)(__m256i)(V), (int)(M)))

#define _mm256_inserti128_si256(V1, V2, M) \
  ((__m256i)__builtin_ia32_insert128i256((__v4di)(__m256i)(V1), \
                                         (__v2di)(__m128i)(V2), (int)(M)))

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskload_epi32(int const *__X, __m256i __M)
{
  return (__m256i)__builtin_ia32_maskloadd256((const __v8si *)__X, (__v8si)__M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskload_epi64(long long const *__X, __m256i __M)
{
  return (__m256i)__builtin_ia32_maskloadq256((const __v4di *)__X, (__v4di)__M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskload_epi32(int const *__X, __m128i __M)
{
  return (__m128i)__builtin_ia32_maskloadd((const __v4si *)__X, (__v4si)__M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskload_epi64(long long const *__X, __m128i __M)
{
  return (__m128i)__builtin_ia32_maskloadq((const __v2di *)__X, (__v2di)__M);
}

static __inline__ void __DEFAULT_FN_ATTRS256
_mm256_maskstore_epi32(int *__X, __m256i __M, __m256i __Y)
{
  __builtin_ia32_maskstored256((__v8si *)__X, (__v8si)__M, (__v8si)__Y);
}

static __inline__ void __DEFAULT_FN_ATTRS256
_mm256_maskstore_epi64(long long *__X, __m256i __M, __m256i __Y)
{
  __builtin_ia32_maskstoreq256((__v4di *)__X, (__v4di)__M, (__v4di)__Y);
}

static __inline__ void __DEFAULT_FN_ATTRS128
_mm_maskstore_epi32(int *__X, __m128i __M, __m128i __Y)
{
  __builtin_ia32_maskstored((__v4si *)__X, (__v4si)__M, (__v4si)__Y);
}

static __inline__ void __DEFAULT_FN_ATTRS128
_mm_maskstore_epi64(long long *__X, __m128i __M, __m128i __Y)
{
  __builtin_ia32_maskstoreq(( __v2di *)__X, (__v2di)__M, (__v2di)__Y);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __X
///    left by the number of bits given in the corresponding element of the
///    256-bit vector of [8 x i32] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLVD instruction.
///
/// \param __X
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __Y
///    A 256-bit vector of [8 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sllv_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psllv8si((__v8si)__X, (__v8si)__Y);
}

/// Shifts each 32-bit element of the 128-bit vector of [4 x i32] in \a __X
///    left by the number of bits given in the corresponding element of the
///    128-bit vector of [4 x i32] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLVD instruction.
///
/// \param __X
///    A 128-bit vector of [4 x i32] to be shifted.
/// \param __Y
///    A 128-bit vector of [4 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 128-bit vector of [4 x i32] containing the result.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_sllv_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psllv4si((__v4si)__X, (__v4si)__Y);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __X
///    left by the number of bits given in the corresponding element of the
///    128-bit vector of [4 x i64] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    63, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLVQ instruction.
///
/// \param __X
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __Y
///    A 256-bit vector of [4 x i64] containing the unsigned shift counts (in
///    bits).
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_sllv_epi64(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psllv4di((__v4di)__X, (__v4di)__Y);
}

/// Shifts each 64-bit element of the 128-bit vector of [2 x i64] in \a __X
///    left by the number of bits given in the corresponding element of the
///    128-bit vector of [2 x i64] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    63, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSLLVQ instruction.
///
/// \param __X
///    A 128-bit vector of [2 x i64] to be shifted.
/// \param __Y
///    A 128-bit vector of [2 x i64] containing the unsigned shift counts (in
///    bits).
/// \returns A 128-bit vector of [2 x i64] containing the result.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_sllv_epi64(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psllv2di((__v2di)__X, (__v2di)__Y);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __X
///    right by the number of bits given in the corresponding element of the
///    256-bit vector of [8 x i32] in \a __Y, shifting in sign bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is 0 or -1 according to the sign bit
///    for that element.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAVD instruction.
///
/// \param __X
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __Y
///    A 256-bit vector of [8 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srav_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrav8si((__v8si)__X, (__v8si)__Y);
}

/// Shifts each 32-bit element of the 128-bit vector of [4 x i32] in \a __X
///    right by the number of bits given in the corresponding element of the
///    128-bit vector of [4 x i32] in \a __Y, shifting in sign bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is 0 or -1 according to the sign bit
///    for that element.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRAVD instruction.
///
/// \param __X
///    A 128-bit vector of [4 x i32] to be shifted.
/// \param __Y
///    A 128-bit vector of [4 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 128-bit vector of [4 x i32] containing the result.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_srav_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrav4si((__v4si)__X, (__v4si)__Y);
}

/// Shifts each 32-bit element of the 256-bit vector of [8 x i32] in \a __X
///    right by the number of bits given in the corresponding element of the
///    256-bit vector of [8 x i32] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLVD instruction.
///
/// \param __X
///    A 256-bit vector of [8 x i32] to be shifted.
/// \param __Y
///    A 256-bit vector of [8 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 256-bit vector of [8 x i32] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srlv_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrlv8si((__v8si)__X, (__v8si)__Y);
}

/// Shifts each 32-bit element of the 128-bit vector of [4 x i32] in \a __X
///    right by the number of bits given in the corresponding element of the
///    128-bit vector of [4 x i32] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    31, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLVD instruction.
///
/// \param __X
///    A 128-bit vector of [4 x i32] to be shifted.
/// \param __Y
///    A 128-bit vector of [4 x i32] containing the unsigned shift counts (in
///    bits).
/// \returns A 128-bit vector of [4 x i32] containing the result.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_srlv_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrlv4si((__v4si)__X, (__v4si)__Y);
}

/// Shifts each 64-bit element of the 256-bit vector of [4 x i64] in \a __X
///    right by the number of bits given in the corresponding element of the
///    128-bit vector of [4 x i64] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    63, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLVQ instruction.
///
/// \param __X
///    A 256-bit vector of [4 x i64] to be shifted.
/// \param __Y
///    A 256-bit vector of [4 x i64] containing the unsigned shift counts (in
///    bits).
/// \returns A 256-bit vector of [4 x i64] containing the result.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_srlv_epi64(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrlv4di((__v4di)__X, (__v4di)__Y);
}

/// Shifts each 64-bit element of the 128-bit vector of [2 x i64] in \a __X
///    right by the number of bits given in the corresponding element of the
///    128-bit vector of [2 x i64] in \a __Y, shifting in zero bits, and
///    returns the result. If the shift count for any element is greater than
///    63, the result for that element is zero.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPSRLVQ instruction.
///
/// \param __X
///    A 128-bit vector of [2 x i64] to be shifted.
/// \param __Y
///    A 128-bit vector of [2 x i64] containing the unsigned shift counts (in
///    bits).
/// \returns A 128-bit vector of [2 x i64] containing the result.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_srlv_epi64(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrlv2di((__v2di)__X, (__v2di)__Y);
}

/// Conditionally gathers two 64-bit floating-point values, either from the
///    128-bit vector of [2 x double] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 128-bit vector
///    of [2 x double] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*32
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128d _mm_mask_i32gather_pd(__m128d a, const double *m, __m128i i,
///                               __m128d mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPD instruction.
///
/// \param a
///    A 128-bit vector of [2 x double] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m. Only
///    the first two elements are used.
/// \param mask
///    A 128-bit vector of [2 x double] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x double] containing the gathered values.
#define _mm_mask_i32gather_pd(a, m, i, mask, s) \
  ((__m128d)__builtin_ia32_gatherd_pd((__v2df)(__m128i)(a), \
                                      (double const *)(m), \
                                      (__v4si)(__m128i)(i), \
                                      (__v2df)(__m128d)(mask), (s)))

/// Conditionally gathers four 64-bit floating-point values, either from the
///    256-bit vector of [4 x double] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 256-bit vector
///    of [4 x double] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*32
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256d _mm256_mask_i32gather_pd(__m256d a, const double *m, __m128i i,
///                                  __m256d mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPD instruction.
///
/// \param a
///    A 256-bit vector of [4 x double] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [4 x double] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x double] containing the gathered values.
#define _mm256_mask_i32gather_pd(a, m, i, mask, s) \
  ((__m256d)__builtin_ia32_gatherd_pd256((__v4df)(__m256d)(a), \
                                         (double const *)(m), \
                                         (__v4si)(__m128i)(i), \
                                         (__v4df)(__m256d)(mask), (s)))

/// Conditionally gathers two 64-bit floating-point values, either from the
///    128-bit vector of [2 x double] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i. The 128-bit vector
///    of [2 x double] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*64
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128d _mm_mask_i64gather_pd(__m128d a, const double *m, __m128i i,
///                               __m128d mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPD instruction.
///
/// \param a
///    A 128-bit vector of [2 x double] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [2 x double] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x double] containing the gathered values.
#define _mm_mask_i64gather_pd(a, m, i, mask, s) \
  ((__m128d)__builtin_ia32_gatherq_pd((__v2df)(__m128d)(a), \
                                      (double const *)(m), \
                                      (__v2di)(__m128i)(i), \
                                      (__v2df)(__m128d)(mask), (s)))

/// Conditionally gathers four 64-bit floating-point values, either from the
///    256-bit vector of [4 x double] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i. The 256-bit vector
///    of [4 x double] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*64
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256d _mm256_mask_i64gather_pd(__m256d a, const double *m, __m256i i,
///                                  __m256d mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPD instruction.
///
/// \param a
///    A 256-bit vector of [4 x double] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [4 x double] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x double] containing the gathered values.
#define _mm256_mask_i64gather_pd(a, m, i, mask, s) \
  ((__m256d)__builtin_ia32_gatherq_pd256((__v4df)(__m256d)(a), \
                                         (double const *)(m), \
                                         (__v4di)(__m256i)(i), \
                                         (__v4df)(__m256d)(mask), (s)))

/// Conditionally gathers four 32-bit floating-point values, either from the
///    128-bit vector of [4 x float] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 128-bit vector
///    of [4 x float] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*32
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm_mask_i32gather_ps(__m128 a, const float *m, __m128i i,
///                              __m128 mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPS instruction.
///
/// \param a
///    A 128-bit vector of [4 x float] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x float] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm_mask_i32gather_ps(a, m, i, mask, s) \
  ((__m128)__builtin_ia32_gatherd_ps((__v4sf)(__m128)(a), \
                                     (float const *)(m), \
                                     (__v4si)(__m128i)(i), \
                                     (__v4sf)(__m128)(mask), (s)))

/// Conditionally gathers eight 32-bit floating-point values, either from the
///    256-bit vector of [8 x float] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [8 x i32] in \a i. The 256-bit vector
///    of [8 x float] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 7
///   j := element*32
///   k := element*32
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256 _mm256_mask_i32gather_ps(__m256 a, const float *m, __m256i i,
///                                 __m256 mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPS instruction.
///
/// \param a
///    A 256-bit vector of [8 x float] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [8 x i32] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [8 x float] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [8 x float] containing the gathered values.
#define _mm256_mask_i32gather_ps(a, m, i, mask, s) \
  ((__m256)__builtin_ia32_gatherd_ps256((__v8sf)(__m256)(a), \
                                        (float const *)(m), \
                                        (__v8si)(__m256i)(i), \
                                        (__v8sf)(__m256)(mask), (s)))

/// Conditionally gathers two 32-bit floating-point values, either from the
///    128-bit vector of [4 x float] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i. The 128-bit vector
///    of [4 x float] in \a mask determines the source for the lower two
///    elements. The upper two elements of the result are zeroed.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*32
///   k := element*64
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// result[127:64] := 0
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm_mask_i64gather_ps(__m128 a, const float *m, __m128i i,
///                              __m128 mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPS instruction.
///
/// \param a
///    A 128-bit vector of [4 x float] used as the source when a mask bit is
///    zero. Only the first two elements are used.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x float] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory. Only the first
///    two elements are used.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm_mask_i64gather_ps(a, m, i, mask, s) \
  ((__m128)__builtin_ia32_gatherq_ps((__v4sf)(__m128)(a), \
                                     (float const *)(m), \
                                     (__v2di)(__m128i)(i), \
                                     (__v4sf)(__m128)(mask), (s)))

/// Conditionally gathers four 32-bit floating-point values, either from the
///    128-bit vector of [4 x float] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i. The 128-bit vector
///    of [4 x float] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*64
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm256_mask_i64gather_ps(__m128 a, const float *m, __m256i i,
///                                 __m128 mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPS instruction.
///
/// \param a
///    A 128-bit vector of [4 x float] used as the source when a mask bit is
///   zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x float] containing the mask. The most
///    significant bit of each element in the mask vector represents the mask
///    bits. If a mask bit is zero, the corresponding value from vector \a a
///    is gathered; otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm256_mask_i64gather_ps(a, m, i, mask, s) \
  ((__m128)__builtin_ia32_gatherq_ps256((__v4sf)(__m128)(a), \
                                        (float const *)(m), \
                                        (__v4di)(__m256i)(i), \
                                        (__v4sf)(__m128)(mask), (s)))

/// Conditionally gathers four 32-bit integer values, either from the
///    128-bit vector of [4 x i32] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 128-bit vector
///    of [4 x i32] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*32
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_mask_i32gather_epi32(__m128i a, const int *m, __m128i i,
///                                  __m128i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDD instruction.
///
/// \param a
///    A 128-bit vector of [4 x i32] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x i32] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm_mask_i32gather_epi32(a, m, i, mask, s) \
  ((__m128i)__builtin_ia32_gatherd_d((__v4si)(__m128i)(a), \
                                     (int const *)(m), \
                                     (__v4si)(__m128i)(i), \
                                     (__v4si)(__m128i)(mask), (s)))

/// Conditionally gathers eight 32-bit integer values, either from the
///    256-bit vector of [8 x i32] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [8 x i32] in \a i. The 256-bit vector
///    of [8 x i32] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 7
///   j := element*32
///   k := element*32
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_mask_i32gather_epi32(__m256i a, const int *m, __m256i i,
///                                     __m256i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDD instruction.
///
/// \param a
///    A 256-bit vector of [8 x i32] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [8 x i32] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [8 x i32] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [8 x i32] containing the gathered values.
#define _mm256_mask_i32gather_epi32(a, m, i, mask, s) \
  ((__m256i)__builtin_ia32_gatherd_d256((__v8si)(__m256i)(a), \
                                        (int const *)(m), \
                                        (__v8si)(__m256i)(i), \
                                        (__v8si)(__m256i)(mask), (s)))

/// Conditionally gathers two 32-bit integer values, either from the
///    128-bit vector of [4 x i32] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i. The 128-bit vector
///    of [4 x i32] in \a mask determines the source for the lower two
///    elements. The upper two elements of the result are zeroed.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*32
///   k := element*64
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// result[127:64] := 0
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_mask_i64gather_epi32(__m128i a, const int *m, __m128i i,
///                                  __m128i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQD instruction.
///
/// \param a
///    A 128-bit vector of [4 x i32] used as the source when a mask bit is
///   zero. Only the first two elements are used.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x i32] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory. Only the first two elements
///    are used.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm_mask_i64gather_epi32(a, m, i, mask, s) \
  ((__m128i)__builtin_ia32_gatherq_d((__v4si)(__m128i)(a), \
                                     (int const *)(m), \
                                     (__v2di)(__m128i)(i), \
                                     (__v4si)(__m128i)(mask), (s)))

/// Conditionally gathers four 32-bit integer values, either from the
///    128-bit vector of [4 x i32] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i. The 128-bit vector
///    of [4 x i32] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*64
///   IF mask[j+31] == 0
///     result[j+31:j] := a[j+31:j]
///   ELSE
///     result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm256_mask_i64gather_epi32(__m128i a, const int *m, __m256i i,
///                                     __m128i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQD instruction.
///
/// \param a
///    A 128-bit vector of [4 x i32] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [4 x i32] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm256_mask_i64gather_epi32(a, m, i, mask, s) \
  ((__m128i)__builtin_ia32_gatherq_d256((__v4si)(__m128i)(a), \
                                        (int const *)(m), \
                                        (__v4di)(__m256i)(i), \
                                        (__v4si)(__m128i)(mask), (s)))

/// Conditionally gathers two 64-bit integer values, either from the
///    128-bit vector of [2 x i64] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 128-bit vector
///    of [2 x i64] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*32
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_mask_i32gather_epi64(__m128i a, const long long *m, __m128i i,
///                                  __m128i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDQ instruction.
///
/// \param a
///    A 128-bit vector of [2 x i64] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m. Only
///    the first two elements are used.
/// \param mask
///    A 128-bit vector of [2 x i64] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x i64] containing the gathered values.
#define _mm_mask_i32gather_epi64(a, m, i, mask, s) \
  ((__m128i)__builtin_ia32_gatherd_q((__v2di)(__m128i)(a), \
                                     (long long const *)(m), \
                                     (__v4si)(__m128i)(i), \
                                     (__v2di)(__m128i)(mask), (s)))

/// Conditionally gathers four 64-bit integer values, either from the
///    256-bit vector of [4 x i64] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i. The 256-bit vector
///    of [4 x i64] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*32
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_mask_i32gather_epi64(__m256i a, const long long *m,
///                                     __m128i i, __m256i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDQ instruction.
///
/// \param a
///    A 256-bit vector of [4 x i64] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [4 x i64] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x i64] containing the gathered values.
#define _mm256_mask_i32gather_epi64(a, m, i, mask, s) \
  ((__m256i)__builtin_ia32_gatherd_q256((__v4di)(__m256i)(a), \
                                        (long long const *)(m), \
                                        (__v4si)(__m128i)(i), \
                                        (__v4di)(__m256i)(mask), (s)))

/// Conditionally gathers two 64-bit integer values, either from the
///    128-bit vector of [2 x i64] in \a a, or from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i. The 128-bit vector
///    of [2 x i64] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*64
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_mask_i64gather_epi64(__m128i a, const long long *m, __m128i i,
///                                  __m128i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQQ instruction.
///
/// \param a
///    A 128-bit vector of [2 x i64] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param mask
///    A 128-bit vector of [2 x i64] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x i64] containing the gathered values.
#define _mm_mask_i64gather_epi64(a, m, i, mask, s) \
  ((__m128i)__builtin_ia32_gatherq_q((__v2di)(__m128i)(a), \
                                     (long long const *)(m), \
                                     (__v2di)(__m128i)(i), \
                                     (__v2di)(__m128i)(mask), (s)))

/// Conditionally gathers four 64-bit integer values, either from the
///    256-bit vector of [4 x i64] in \a a, or from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i. The 256-bit vector
///    of [4 x i64] in \a mask determines the source for each element.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*64
///   IF mask[j+63] == 0
///     result[j+63:j] := a[j+63:j]
///   ELSE
///     result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
///   FI
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_mask_i64gather_epi64(__m256i a, const long long *m,
///                                     __m256i i, __m256i mask, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQQ instruction.
///
/// \param a
///    A 256-bit vector of [4 x i64] used as the source when a mask bit is
///    zero.
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param mask
///    A 256-bit vector of [4 x i64] containing the mask. The most significant
///    bit of each element in the mask vector represents the mask bits. If a
///    mask bit is zero, the corresponding value from vector \a a is gathered;
///    otherwise the value is loaded from memory.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x i64] containing the gathered values.
#define _mm256_mask_i64gather_epi64(a, m, i, mask, s) \
  ((__m256i)__builtin_ia32_gatherq_q256((__v4di)(__m256i)(a), \
                                        (long long const *)(m), \
                                        (__v4di)(__m256i)(i), \
                                        (__v4di)(__m256i)(mask), (s)))

/// Gathers two 64-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*32
///   result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128d _mm_i32gather_pd(const double *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m. Only
///    the first two elements are used.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x double] containing the gathered values.
#define _mm_i32gather_pd(m, i, s) \
  ((__m128d)__builtin_ia32_gatherd_pd((__v2df)_mm_undefined_pd(), \
                                      (double const *)(m), \
                                      (__v4si)(__m128i)(i), \
                                      (__v2df)_mm_cmpeq_pd(_mm_setzero_pd(), \
                                                           _mm_setzero_pd()), \
                                      (s)))

/// Gathers four 64-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*32
///   result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256d _mm256_i32gather_pd(const double *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x double] containing the gathered values.
#define _mm256_i32gather_pd(m, i, s) \
  ((__m256d)__builtin_ia32_gatherd_pd256((__v4df)_mm256_undefined_pd(), \
                                         (double const *)(m), \
                                         (__v4si)(__m128i)(i), \
                                         (__v4df)_mm256_cmp_pd(_mm256_setzero_pd(), \
                                                               _mm256_setzero_pd(), \
                                                               _CMP_EQ_OQ), \
                                         (s)))

/// Gathers two 64-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*64
///   result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128d _mm_i64gather_pd(const double *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x double] containing the gathered values.
#define _mm_i64gather_pd(m, i, s) \
  ((__m128d)__builtin_ia32_gatherq_pd((__v2df)_mm_undefined_pd(), \
                                      (double const *)(m), \
                                      (__v2di)(__m128i)(i), \
                                      (__v2df)_mm_cmpeq_pd(_mm_setzero_pd(), \
                                                           _mm_setzero_pd()), \
                                      (s)))

/// Gathers four 64-bit floating-point values from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*64
///   result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256d _mm256_i64gather_pd(const double *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x double] containing the gathered values.
#define _mm256_i64gather_pd(m, i, s) \
  ((__m256d)__builtin_ia32_gatherq_pd256((__v4df)_mm256_undefined_pd(), \
                                         (double const *)(m), \
                                         (__v4di)(__m256i)(i), \
                                         (__v4df)_mm256_cmp_pd(_mm256_setzero_pd(), \
                                                               _mm256_setzero_pd(), \
                                                               _CMP_EQ_OQ), \
                                         (s)))

/// Gathers four 32-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*32
///   result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm_i32gather_ps(const float *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPS instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm_i32gather_ps(m, i, s) \
  ((__m128)__builtin_ia32_gatherd_ps((__v4sf)_mm_undefined_ps(), \
                                     (float const *)(m), \
                                     (__v4si)(__m128i)(i), \
                                     (__v4sf)_mm_cmpeq_ps(_mm_setzero_ps(), \
                                                          _mm_setzero_ps()), \
                                     (s)))

/// Gathers eight 32-bit floating-point values from memory \a m using scaled
///    indexes from the 256-bit vector of [8 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 7
///   j := element*32
///   k := element*32
///   result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256 _mm256_i32gather_ps(const float *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERDPS instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [8 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [8 x float] containing the gathered values.
#define _mm256_i32gather_ps(m, i, s) \
  ((__m256)__builtin_ia32_gatherd_ps256((__v8sf)_mm256_undefined_ps(), \
                                        (float const *)(m), \
                                        (__v8si)(__m256i)(i), \
                                        (__v8sf)_mm256_cmp_ps(_mm256_setzero_ps(), \
                                                              _mm256_setzero_ps(), \
                                                              _CMP_EQ_OQ), \
                                        (s)))

/// Gathers two 32-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [2 x i64] in \a i. The upper two
///    elements of the result are zeroed.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*32
///   k := element*64
///   result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// result[127:64] := 0
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm_i64gather_ps(const float *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPS instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm_i64gather_ps(m, i, s) \
  ((__m128)__builtin_ia32_gatherq_ps((__v4sf)_mm_undefined_ps(), \
                                     (float const *)(m), \
                                     (__v2di)(__m128i)(i), \
                                     (__v4sf)_mm_cmpeq_ps(_mm_setzero_ps(), \
                                                          _mm_setzero_ps()), \
                                     (s)))

/// Gathers four 32-bit floating-point values from memory \a m using scaled
///    indexes from the 256-bit vector of [4 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*64
///   result[j+31:j] := Load32(m + SignExtend(i[k+64:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128 _mm256_i64gather_ps(const float *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VGATHERQPS instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x float] containing the gathered values.
#define _mm256_i64gather_ps(m, i, s) \
  ((__m128)__builtin_ia32_gatherq_ps256((__v4sf)_mm_undefined_ps(), \
                                        (float const *)(m), \
                                        (__v4di)(__m256i)(i), \
                                        (__v4sf)_mm_cmpeq_ps(_mm_setzero_ps(), \
                                                             _mm_setzero_ps()), \
                                        (s)))

/// Gathers four 32-bit floating-point values from memory \a m using scaled
///    indexes from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*32
///   result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_i32gather_epi32(const int *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm_i32gather_epi32(m, i, s) \
  ((__m128i)__builtin_ia32_gatherd_d((__v4si)_mm_undefined_si128(), \
                                     (int const *)(m), (__v4si)(__m128i)(i), \
                                     (__v4si)_mm_set1_epi32(-1), (s)))

/// Gathers eight 32-bit floating-point values from memory \a m using scaled
///    indexes from the 256-bit vector of [8 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 7
///   j := element*32
///   k := element*32
///   result[j+31:j] := Load32(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_i32gather_epi32(const int *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [8 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [8 x i32] containing the gathered values.
#define _mm256_i32gather_epi32(m, i, s) \
  ((__m256i)__builtin_ia32_gatherd_d256((__v8si)_mm256_undefined_si256(), \
                                        (int const *)(m), (__v8si)(__m256i)(i), \
                                        (__v8si)_mm256_set1_epi32(-1), (s)))

/// Gathers two 32-bit integer values from memory \a m using scaled indexes
///    from the 128-bit vector of [2 x i64] in \a i. The upper two elements
///    of the result are zeroed.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*32
///   k := element*64
///   result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// result[127:64] := 0
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_i64gather_epi32(const int *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm_i64gather_epi32(m, i, s) \
  ((__m128i)__builtin_ia32_gatherq_d((__v4si)_mm_undefined_si128(), \
                                     (int const *)(m), (__v2di)(__m128i)(i), \
                                     (__v4si)_mm_set1_epi32(-1), (s)))

/// Gathers four 32-bit integer values from memory \a m using scaled indexes
///    from the 256-bit vector of [4 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*32
///   k := element*64
///   result[j+31:j] := Load32(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm256_i64gather_epi32(const int *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQD instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [4 x i32] containing the gathered values.
#define _mm256_i64gather_epi32(m, i, s) \
  ((__m128i)__builtin_ia32_gatherq_d256((__v4si)_mm_undefined_si128(), \
                                        (int const *)(m), (__v4di)(__m256i)(i), \
                                        (__v4si)_mm_set1_epi32(-1), (s)))

/// Gathers two 64-bit integer values from memory \a m using scaled indexes
///    from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*32
///   result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_i32gather_epi64(const long long *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDQ instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m. Only
///    the first two elements are used.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x i64] containing the gathered values.
#define _mm_i32gather_epi64(m, i, s) \
  ((__m128i)__builtin_ia32_gatherd_q((__v2di)_mm_undefined_si128(), \
                                     (long long const *)(m), \
                                     (__v4si)(__m128i)(i), \
                                     (__v2di)_mm_set1_epi64x(-1), (s)))

/// Gathers four 64-bit integer values from memory \a m using scaled indexes
///    from the 128-bit vector of [4 x i32] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*32
///   result[j+63:j] := Load64(m + SignExtend(i[k+31:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_i32gather_epi64(const long long *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERDQ instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [4 x i32] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x i64] containing the gathered values.
#define _mm256_i32gather_epi64(m, i, s) \
  ((__m256i)__builtin_ia32_gatherd_q256((__v4di)_mm256_undefined_si256(), \
                                        (long long const *)(m), \
                                        (__v4si)(__m128i)(i), \
                                        (__v4di)_mm256_set1_epi64x(-1), (s)))

/// Gathers two 64-bit integer values from memory \a m using scaled indexes
///    from the 128-bit vector of [2 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 1
///   j := element*64
///   k := element*64
///   result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m128i _mm_i64gather_epi64(const long long *m, __m128i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQQ instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 128-bit vector of [2 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 128-bit vector of [2 x i64] containing the gathered values.
#define _mm_i64gather_epi64(m, i, s) \
  ((__m128i)__builtin_ia32_gatherq_q((__v2di)_mm_undefined_si128(), \
                                     (long long const *)(m), \
                                     (__v2di)(__m128i)(i), \
                                     (__v2di)_mm_set1_epi64x(-1), (s)))

/// Gathers four 64-bit integer values from memory \a m using scaled indexes
///    from the 256-bit vector of [4 x i64] in \a i.
///
/// \code{.operation}
/// FOR element := 0 to 3
///   j := element*64
///   k := element*64
///   result[j+63:j] := Load64(m + SignExtend(i[k+63:k])*s)
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// \code
/// __m256i _mm256_i64gather_epi64(const long long *m, __m256i i, const int s);
/// \endcode
///
/// This intrinsic corresponds to the \c VPGATHERQQ instruction.
///
/// \param m
///    A pointer to the memory used for loading values.
/// \param i
///    A 256-bit vector of [4 x i64] containing signed indexes into \a m.
/// \param s
///    A literal constant scale factor for the indexes in \a i. Must be
///    1, 2, 4, or 8.
/// \returns A 256-bit vector of [4 x i64] containing the gathered values.
#define _mm256_i64gather_epi64(m, i, s) \
  ((__m256i)__builtin_ia32_gatherq_q256((__v4di)_mm256_undefined_si256(), \
                                        (long long const *)(m), \
                                        (__v4di)(__m256i)(i), \
                                        (__v4di)_mm256_set1_epi64x(-1), (s)))

#undef __DEFAULT_FN_ATTRS256
#undef __DEFAULT_FN_ATTRS128

#endif /* __AVX2INTRIN_H */
