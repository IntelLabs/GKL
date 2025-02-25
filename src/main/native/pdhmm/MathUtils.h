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
#ifndef _MATHUTILS_H
#define _MATHUTILS_H
#include <cstdint>
#include <immintrin.h>

extern double INITIAL_CONDITION;
extern double INITIAL_CONDITION_LOG10;
extern const double neg_infinity;
extern const double LN10;
extern const double INV_LN10;

class JacobianLogTable
{
public:
    static JacobianLogTable &getInstance()
    {
        static JacobianLogTable instance;
        return instance;
    }

    double get(double difference);
    void initCache();
    void freeCache();

    static double TABLE_STEP;
    static double INV_STEP;
    static double MAX_TOLERANCE;
    static double *cache;

private:
    JacobianLogTable() = default;
    ~JacobianLogTable() = default;
    JacobianLogTable(const JacobianLogTable &) = delete;
    JacobianLogTable &operator=(const JacobianLogTable &) = delete;

    double cacheIntToDouble(int i);
};

double approximateLog10SumLog10(double a, double b);
int32_t fastRound(double d);
bool isValidLog10Probability(double result);

enum PartiallyDeterminedHaplotype
{
    SNP = 1,
    DEL_START = 2,
    DEL_END = 4,
    A = 8,
    C = 16,
    G = 32,
    T = 64,
};

#endif
