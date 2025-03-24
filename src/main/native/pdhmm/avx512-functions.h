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
#ifndef _AVX512_FUNCTIONS_H
#define _AVX512_FUNCTIONS_H

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
#undef VEC_BLEND_INT
#undef VEC_BLEND_PD_DOUBLE_MASK
#undef VEC_BLEND_LONG_DOUBLE_MASK
#undef VEC_BLEND_PD

#undef VEC_CMP_EQ_INT
#undef VEC_CMP_EQ_PD
#undef VEC_CMP_GT_PD
#undef VEC_CMP_LE_INT
#undef VEC_CMP_NE_INT
#undef VEC_CMP_LT_INT
#undef VEC_CMP_LT_PD
#undef VEC_CMP_NE_MASK
#undef VEC_CMP_NE_DOUBLE_MASK
#undef VEC_CVT_INT_PD
#undef VEC_CVT_ROUND_PD_LONG
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

#define VEC_DOUBLE_TYPE __m512d
#define VEC_INT_TYPE __m512i
#define INT_TYPE int64_t
#define IS_INT32 0
#define VEC_LONG_TYPE __m512i
#define VEC_MASK_TYPE __mmask8
#define VEC_DOUBLE_MASK_TYPE __mmask8
#define SIMD_WIDTH_DOUBLE 8

// Primary Macros

#define VEC_ADD_INT(__v1, __v2) \
    _mm512_add_epi64(__v1, __v2)

#define VEC_ADD_PD(__v1, __v2) \
    _mm512_add_pd(__v1, __v2)

#define VEC_AND_INT(__v1, __v2) \
    _mm512_and_epi64(__v1, __v2)

#define VEC_AND_MASK(__v1, __v2) \
    _kand_mask8(__v1, __v2)

#define VEC_BLEND_INT(__v1, __v2, __mask) \
    _mm512_mask_blend_epi64(__mask, __v1, __v2)

#define VEC_BLEND_LONG_DOUBLE_MASK(__v1, __v2, __mask) \
    _mm512_mask_blend_epi64(__mask, __v1, __v2)

#define VEC_BLEND_PD(__v1, __v2, __mask) \
    _mm512_mask_blend_pd(__mask, __v1, __v2)

#define VEC_BLEND_PD_DOUBLE_MASK(__v1, __v2, __mask) \
    _mm512_mask_blend_pd(__mask, __v1, __v2)

#define VEC_CMP_EQ_INT(__v1, __v2) \
    _mm512_cmp_epi64_mask(__v1, __v2, _MM_CMPINT_EQ)

#define VEC_CMP_EQ_PD(__v1, __v2) \
    _mm512_cmp_pd_mask(__v1, __v2, _CMP_EQ_OQ)

#define VEC_CMP_GT_PD(__v1, __v2) \
    _mm512_cmp_pd_mask(__v1, __v2, _CMP_GT_OQ)

#define VEC_CMP_LE_INT(__v1, __v2) \
    _mm512_cmp_epi64_mask(__v1, __v2, _MM_CMPINT_LE)

#define VEC_CMP_NE_INT(__v1, __v2) \
    _mm512_cmp_epi64_mask(__v1, __v2, _MM_CMPINT_NE)

#define VEC_CMP_LT_INT(__v1, __v2) \
    _mm512_cmp_epi64_mask(__v1, __v2, _MM_CMPINT_LT)

#define VEC_CMP_LT_PD(__v1, __v2) \
    _mm512_cmp_pd_mask(__v1, __v2, _CMP_LT_OS)

#define VEC_CMP_NE_MASK(__mask, __value) \
    ((__mask) != (__value))

#define VEC_CMP_NE_DOUBLE_MASK(__mask, __value) \
    ((__mask) != (__value))

#define VEC_CVT_INT_PD(__v1) \
    _mm512_cvtepi64_pd(__v1)

#define VEC_CVT_ROUND_PD_LONG(__v1) \
    _mm512_cvt_roundpd_epi64(__v1, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC))

#define VEC_DIV_PD(__v1, __v2) \
    _mm512_div_pd(__v1, __v2)

#define VEC_GATHER_PD(__index, __array, __scale) \
    _mm512_i64gather_pd(__index, __array, __scale)

#define VEC_GATHER_PD_LONG_INDEX(__index, __array, __scale) \
    _mm512_i64gather_pd(__index, __array, __scale)

#define VEC_LOAD_INT(__v1) \
    _mm512_load_epi64(__v1)

#define VEC_LOAD_PD(__v1) \
    _mm512_load_pd(__v1)

#define VEC_LOADU_INT(__v1) \
    _mm512_loadu_si512(__v1)

#define VEC_MAX_INT(__v1, __v2) \
    _mm512_max_epi64(__v1, __v2)

#define VEC_MAX_PD(__v1, __v2) \
    _mm512_max_pd(__v1, __v2)

#define VEC_MIN_INT(__v1, __v2) \
    _mm512_min_epi64(__v1, __v2)

#define VEC_MIN_PD(__v1, __v2) \
    _mm512_min_pd(__v1, __v2)

#define VEC_MUL_PD(__v1, __v2) \
    _mm512_mul_pd(__v1, __v2)

#define VEC_MULLO_INT(__v1, __v2) \
    _mm512_mullo_epi64(__v1, __v2)

#define VEC_OR_MASK(__v1, __v2) \
    _kor_mask8(__v1, __v2)

#define VEC_SET0_INT() \
    _mm512_setzero_si512()

#define VEC_SET0_PD() \
    _mm512_setzero_pd()

#define VEC_SET1_INT(__v1) \
    _mm512_set1_epi64(__v1)

#define VEC_SET1_PD(__v1) \
    _mm512_set1_pd(__v1)

#define VEC_SRLI_INT(__v1, __v2) \
    _mm512_srli_epi64(__v1, __v2)

#define VEC_STORE_PD(__v1, __v2) \
    _mm512_store_pd(__v1, __v2)

#define VEC_STORE_INT(__v1, __v2) \
    _mm512_store_si512((VEC_INT_TYPE *)__v1, __v2)

#define VEC_SUB_INT(__v1, __v2) \
    _mm512_sub_epi64(__v1, __v2)

#define VEC_SUB_PD(__v1, __v2) \
    _mm512_sub_pd(__v1, __v2)

#endif
