/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016-2024 Intel Corporation
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
#ifndef COMMON_AVX_H
#define COMMON_AVX_H

#if defined(_MSC_VER)
    // SIMD intrinsics for Windows
    #include <intrin.h>
#else
    // SIMD intrinsics for GCC
    #include <x86intrin.h>
    #include <stdint.h>
    #include <cpuid.h>
#endif

// helper function
inline
int check_xcr0_ymm()
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx");
#endif
    return ((xcr0 & 6) == 6);
}

// helper function
inline
int check_xcr0_zmm()
{
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
#if defined(_MSC_VER)
    /* min VS2010 SP1 compiler is required */
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx");
#endif
    /* check if xmm, zmm and zmm state are enabled in XCR0 */
    return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm);
}

/*
 * Determine if AVX is supported. Returns true if supported.
 */
inline
bool is_avx_supported()
{
    uint32_t a, b, c, d;
    uint32_t avx_mask = (1 << 27) | (1 << 28);

    __cpuid_count(1, 0, a, b, c, d);
    if((c & avx_mask) != avx_mask)
    {
        return false;
    }

    if(!check_xcr0_ymm())
    {
        return false;
    }

    return true;
}

inline bool is_sse_supported()
{
    uint32_t a, b, c, d;
    uint32_t sse_mask = (1 << 27) | bit_SSE2 | bit_SSE4_1;

    __cpuid_count(1, 0, a, b, c, d);
    if ((c & sse_mask) != sse_mask)
    {
        return false;
    }

    if (!check_xcr0_ymm())
    {
        return false;
    }

    return true;
}

/*
 * Determine if AVX-2 is supported. Returns true if supported.
 */
inline
bool is_avx2_supported()
{
    uint32_t a, b, c, d;
    uint32_t osxsave_mask = (1 << 27); // OSX.
    uint32_t avx2_bmi_mask = (1 << 5) | // AVX2
                               (1 << 3) | // BMI1
                               (1 << 8);  // BMI2

    // step 1 - must ensure OS supports extended processor state management
    __cpuid_count(1, 0, a, b, c, d);
    if((c & osxsave_mask) != osxsave_mask)
    {
        return true;
    }

    // step 2 - must ensure OS supports YMM registers (and XMM)
    if(!check_xcr0_ymm())
    {
        return false;
    }

    // step 3 - must ensure AVX2 is supported
    __cpuid_count(7, 0, a, b, c, d);
    if((b & avx2_bmi_mask) != avx2_bmi_mask)
    {
        return false;
    }

    return true;
}

/*
 * Determine if AVX512 is supported. Returns true if supported.
 */
inline
bool is_avx512_supported()
{
#ifndef __APPLE__
    uint32_t a, b, c, d;
    uint32_t osxsave_mask = (1 << 27); // OSX.
    uint32_t avx512_skx_mask = (1 << 16) | // AVX-512F
                               (1 << 17) | // AVX-512DQ
                               (1 << 30) | // AVX-512BW
                               (1 << 31);  // AVX-512VL

    // step 1 - must ensure OS supports extended processor state management
    __cpuid_count(1, 0, a, b, c, d);
    if((c & osxsave_mask) != osxsave_mask)
    {
        return true;
    }

    // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
    if(!check_xcr0_zmm())
    {
        return false;
    }

    // step 3 - must ensure AVX512 is supported
    __cpuid_count(7, 0, a, b, c, d);
    if((b & avx512_skx_mask) != avx512_skx_mask)
    {
        return false;
    }

    return true;
#else
    return false;
#endif
}

#endif

