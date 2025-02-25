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
#include "MathUtils.h"
#include <cmath>
#include <immintrin.h>
#include <unistd.h>

using namespace std;

double INITIAL_CONDITION = pow(2, 1020);
double INITIAL_CONDITION_LOG10 = log10(INITIAL_CONDITION);
const double neg_infinity = -1e10;
double JacobianLogTable::TABLE_STEP = 0.0001;
double JacobianLogTable::INV_STEP = 1.0 / TABLE_STEP;
double JacobianLogTable::MAX_TOLERANCE = 8.0;
double *JacobianLogTable::cache = NULL;
const double LN10 = log(10);
const double INV_LN10 = 1.0 / LN10;

int32_t fastRound(double d)
{
    return (d > 0.0) ? (int32_t)(d + 0.5) : (int32_t)(d - 0.5);
}

bool isValidLog10Probability(double result) { return result <= 0.0; }

double JacobianLogTable::get(double difference)
{
    int index = fastRound(difference * INV_STEP);
    return cache[index];
}

void JacobianLogTable::initCache()
{
    if (cache != NULL)
    {
        // If cache is already initialized, free it first
        freeCache();
    }

    int from = 0;
    int to = (int)(MAX_TOLERANCE / TABLE_STEP) + 1;
    double *result = (double *)_mm_malloc((to - from) * sizeof(double), 64);
    // if (result == NULL)
    // {
    //     throw std::bad_alloc();
    // }
    for (int i = from; i < to; i++)
    {
        result[i - from] = cacheIntToDouble(i);
    }
    cache = result;
}

void JacobianLogTable::freeCache()
{
    if (cache != NULL)
    {
        _mm_free(cache);
        cache = NULL;
    }
}

double JacobianLogTable::cacheIntToDouble(int k)
{
    return log10(1.0 + pow(10.0, -k * TABLE_STEP));
}

double approximateLog10SumLog10(double a, double b)
{
    // this code works only when a <= b so we flip them if the order is opposite
    if (a > b)
    {
        return approximateLog10SumLog10(b, a);
    }
    else if (a == neg_infinity)
    {
        return b;
    }

    // if |b-a| < tol we need to compute log(e^a + e^b) = log(e^b(1 + e^(a-b))) =
    // b + log(1 + e^(-(b-a))) we compute the second term as a table lookup with
    // integer quantization we have pre-stored correction for 0,0.1,0.2,... 10.0
    double diff = b - a;
    return b + (diff < JacobianLogTable::MAX_TOLERANCE
                    ? JacobianLogTable::getInstance().get(diff)
                    : 0.0);
}
