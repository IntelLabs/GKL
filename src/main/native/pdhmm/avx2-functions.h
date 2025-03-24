/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023-2024 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _AVX2_FUNCTIONS_H
#define _AVX2_FUNCTIONS_H

#undef VEC_DOUBLE_TYPE
#undef VEC_INT_TYPE
#undef INT_TYPE
#undef IS_INT32
#undef VEC_LONG_TYPE
#undef VEC_MASK_TYPE
#undef VEC_DOUBLE_MASK_TYPE
#undef SIMD_WIDTH_DOUBLE

#undef VEC_ADD_INT
#undef VEC_ADD_PD
#undef VEC_AND_INT
#undef VEC_AND_MASK
#undef VEC_AND_NOT_INT
#undef VEC_BLEND_INT
#undef VEC_BLEND_PD_DOUBLE_MASK
#undef VEC_CAST_INT_PD
#undef VEC_CAST_PD_LONG
#undef VEC_CMP_EQ_INT
#undef VEC_CMP_EQ_PD
#undef VEC_CMP_GT_PD
#undef VEC_CMP_LT_INT
#undef VEC_CMP_LT_PD
#undef VEC_CMP_NE_MASK
#undef VEC_CMP_NE_DOUBLE_MASK
#undef VEC_CVT_INT_LONG
#undef VEC_CVT_INT_PD
#undef VEC_CVT_PD_INT
#undef VEC_DIV_PD
#undef VEC_GATHER_PD
#undef VEC_GATHER_PD_LONG_INDEX
#undef VEC_LOAD_INT
#undef VEC_LOAD_PD
#undef VEC_LOADU_INT
#undef VEC_MAX_INT
#undef VEC_MAX_PD
#undef VEC_MIN_INT
#undef VEC_MIN_PD
#undef VEC_MUL_PD
#undef VEC_MULLO_INT
#undef VEC_OR_MASK
#undef VEC_SET0_INT
#undef VEC_SET0_PD
#undef VEC_SET1_INT
#undef VEC_SET1_PD
#undef VEC_SRLI_INT
#undef VEC_STORE_PD
#undef VEC_STORE_INT
#undef VEC_SUB_INT
#undef VEC_SUB_PD

#undef VEC_BLEND_LONG_DOUBLE_MASK
#undef VEC_BLEND_PD
#undef VEC_CMP_LE_INT
#undef VEC_CMP_NE_INT
#undef VEC_CVT_ROUND_PD_LONG

#define VEC_DOUBLE_TYPE __m256d
#define VEC_INT_TYPE __m128i
#define INT_TYPE int32_t
#define IS_INT32 1
#define VEC_LONG_TYPE __m256i
#define VEC_MASK_TYPE __m128i
#define VEC_DOUBLE_MASK_TYPE __m256d
#define SIMD_WIDTH_DOUBLE 4

// Primary Macros
#define VEC_ADD_INT(__v1, __v2) \
    _mm_add_epi32(__v1, __v2)

#define VEC_ADD_PD(__v1, __v2) \
    _mm256_add_pd(__v1, __v2)

#define VEC_AND_INT(__v1, __v2) \
    _mm_and_si128(__v1, __v2)

#define VEC_AND_MASK(__v1, __v2) \
    _mm_and_si128(__v1, __v2)

#define VEC_AND_NOT_INT(__v1, __v2) \
    _mm_andnot_si128(__v1, __v2)

#define VEC_BLEND_INT(__v1, __v2, __mask) \
    _mm_blendv_epi8(__v1, __v2, __mask)

#define VEC_BLEND_PD_DOUBLE_MASK(__v1, __v2, __mask) \
    _mm256_blendv_pd(__v1, __v2, __mask)

#define VEC_CAST_INT_PD(__v1) \
    _mm256_castsi256_pd(__v1)

#define VEC_CAST_PD_LONG(__v1) \
    _mm256_castpd_si256(__v1)

#define VEC_CMP_EQ_INT(__v1, __v2) \
    _mm_cmpeq_epi32(__v1, __v2)

#define VEC_CMP_EQ_PD(__v1, __v2) \
    _mm256_cmp_pd(__v1, __v2, _CMP_EQ_OQ)

#define VEC_CMP_GT_PD(__v1, __v2) \
    _mm256_cmp_pd(__v1, __v2, _CMP_GT_OQ)

#define VEC_CMP_LT_INT(__v1, __v2) \
    _mm_cmplt_epi32(__v1, __v2)

#define VEC_CMP_LT_PD(__v1, __v2) \
    _mm256_cmp_pd(__v1, __v2, _CMP_LT_OS)

#define VEC_CMP_NE_MASK(__mask, __value) \
    (_mm_movemask_epi8(__mask) != (__value))

#define VEC_CMP_NE_DOUBLE_MASK(__mask, __value) \
    (_mm256_movemask_pd(__mask) != (__value))

#define VEC_CVT_INT_LONG(__v1) \
    _mm256_cvtepi32_epi64(__v1)

#define VEC_CVT_INT_PD(__v1) \
    _mm256_cvtepi32_pd(__v1)

#define VEC_CVT_PD_INT(__v1) \
    _mm256_cvtpd_epi32(__v1)

#define VEC_DIV_PD(__v1, __v2) \
    _mm256_div_pd(__v1, __v2)

#define VEC_GATHER_PD(__index, __array, __scale) \
    _mm256_i32gather_pd(__array, __index, __scale)

#define VEC_GATHER_PD_LONG_INDEX(__index, __array, __scale) \
    _mm256_i64gather_pd(__array, __index, __scale)

#define VEC_LOAD_INT(__v1) \
    _mm_load_si128((VEC_INT_TYPE *)(__v1))

#define VEC_LOAD_PD(__v1) \
    _mm256_load_pd(__v1)

#define VEC_LOADU_INT(__v1) \
    _mm_loadu_si128((VEC_INT_TYPE *)(__v1))

#define VEC_MAX_INT(__v1, __v2) \
    _mm_max_epi32(__v1, __v2)

#define VEC_MAX_PD(__v1, __v2) \
    _mm256_max_pd(__v1, __v2)

#define VEC_MIN_INT(__v1, __v2) \
    _mm_min_epi32(__v1, __v2)

#define VEC_MIN_PD(__v1, __v2) \
    _mm256_min_pd(__v1, __v2)

#define VEC_MUL_PD(__v1, __v2) \
    _mm256_mul_pd(__v1, __v2)

#define VEC_MULLO_INT(__v1, __v2) \
    _mm_mullo_epi32(__v1, __v2)

#define VEC_OR_MASK(__v1, __v2) \
    _mm_or_si128(__v1, __v2)

#define VEC_SET0_INT() \
    _mm_setzero_si128()

#define VEC_SET0_PD() \
    _mm256_setzero_pd()

#define VEC_SET1_INT(__v1) \
    _mm_set1_epi32(__v1)

#define VEC_SET1_PD(__v1) \
    _mm256_set1_pd(__v1)

#define VEC_SRLI_INT(__v1, __v2) \
    _mm_srli_epi32(__v1, __v2)

#define VEC_STORE_PD(__v1, __v2) \
    _mm256_store_pd(__v1, __v2)

#define VEC_STORE_INT(__v1, __v2) \
    _mm_store_si128((VEC_INT_TYPE *)__v1, __v2)

#define VEC_SUB_INT(__v1, __v2) \
    _mm_sub_epi32(__v1, __v2)

#define VEC_SUB_PD(__v1, __v2) \
    _mm256_sub_pd(__v1, __v2)

// Secondary Macros

#define VEC_BLEND_LONG_DOUBLE_MASK(__v1, __v2, __mask) \
    _mm256_blendv_epi8(__v1, __v2, VEC_CAST_PD_LONG(__mask))

#define VEC_BLEND_PD(__v1, __v2, __mask) \
    _mm256_blendv_pd(__v1, __v2, VEC_CAST_INT_PD(VEC_CVT_INT_LONG(__mask)))

#define VEC_CMP_LE_INT(__v1, __v2) \
    VEC_OR_MASK(VEC_CMP_LT_INT(__v1, __v2), VEC_CMP_EQ_INT(__v1, __v2))

#define VEC_CMP_NE_INT(__v1, __v2) \
    VEC_AND_NOT_INT(VEC_CMP_EQ_INT(__v1, __v2), VEC_SET1_INT(0xFFFFFFFF))

#define VEC_CVT_ROUND_PD_LONG(__v1) \
    VEC_CVT_INT_LONG(VEC_CVT_PD_INT(_mm256_round_pd(__v1, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC))))

#endif
