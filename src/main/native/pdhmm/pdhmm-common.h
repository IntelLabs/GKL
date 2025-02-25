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

#ifndef PDHMM_COMMON_H
#define PDHMM_COMMON_H
#define TRANS_PROB_ARRAY_LENGTH 6
#define MAX_QUAL 254

#define OFF 1

#define ROW_UNROLL 4
#define ALIGN_SIZE 64

#define CAT(X, Y) X##Y
#define CONCAT(X, Y) CAT(X, Y)

#define PDHMM_SUCCESS 0
#define PDHMM_MEMORY_ALLOCATION_FAILED 1
#define PDHMM_INPUT_DATA_ERROR 2
#define PDHMM_FAILURE 3
#define PDHMM_MEMORY_ACCESS_ERROR 4

#include "MathUtils.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <debug.h>
#include <immintrin.h>
#include <algorithm>

enum HMMState
{
    // The regular state
    NORMAL,

    // Indicating that we must be copying the array elements to the right
    INSIDE_DEL,

    // Indicating that we must handle the special case for merging events after
    // the del
    AFTER_DEL,
};

enum ProbIndex
{
    matchToMatch,
    indelToMatch,
    matchToInsertion,
    insertionToInsertion,
    matchToDeletion,
    deletionToDeletion,
};

class PDHMMInputData
{
public:
    PDHMMInputData(const int8_t *hap_bases, const int8_t *hap_pdbases, const int8_t *read_bases, const int8_t *read_qual,
                   const int8_t *read_ins_qual, const int8_t *read_del_qual, const int8_t *gcp, double *result,
                   int64_t t, const int64_t *hap_lengths, const int64_t *read_lengths, int32_t maxReadLength, int32_t maxHaplotypeLength)
        : hap_bases(hap_bases), hap_pdbases(hap_pdbases), read_bases(read_bases), read_qual(read_qual),
          read_ins_qual(read_ins_qual), read_del_qual(read_del_qual), gcp(gcp), result(result),
          t(t), hap_lengths(hap_lengths), read_lengths(read_lengths), maxReadLength(maxReadLength), maxHaplotypeLength(maxHaplotypeLength) {}

    // Getter functions
    const int8_t *getHapBases() const { return hap_bases; }
    const int8_t *getHapPDBases() const { return hap_pdbases; }
    const int8_t *getReadBases() const { return read_bases; }
    const int8_t *getReadQual() const { return read_qual; }
    const int8_t *getReadInsQual() const { return read_ins_qual; }
    const int8_t *getReadDelQual() const { return read_del_qual; }
    const int8_t *getGCP() const { return gcp; }
    double *getResult() const { return result; }
    int64_t getT() const { return t; }
    const int64_t *getHapLengths() const { return hap_lengths; }
    const int64_t *getReadLengths() const { return read_lengths; }
    int32_t getMaxReadLength() const { return maxReadLength; }
    int32_t getMaxHaplotypeLength() const { return maxHaplotypeLength; }

private:
    const int8_t *hap_bases;
    const int8_t *hap_pdbases;
    const int8_t *read_bases;
    const int8_t *read_qual;
    const int8_t *read_ins_qual;
    const int8_t *read_del_qual;
    const int8_t *gcp;
    double *result;
    int64_t t;
    const int64_t *hap_lengths;
    const int64_t *read_lengths;
    int32_t maxReadLength;
    int32_t maxHaplotypeLength;
};

inline double qualToErrorProb(double qual, int32_t &status)
{
    if (qual < 0.0)
    {
        status = PDHMM_INPUT_DATA_ERROR;
        DBG("deletion quality cannot be less than 0 \n");
    }
    return pow(10.0, qual / -10.0);
}

class ProbabilityCache
{
public:
    static ProbabilityCache &getInstance()
    {
        static ProbabilityCache instance;
        return instance;
    }

    int32_t initialize()
    {
        if (initialized)
        {
            return PDHMM_SUCCESS;
        }

        int32_t status = PDHMM_SUCCESS;
        JacobianLogTable &jacobianLogTable = JacobianLogTable::getInstance();
        jacobianLogTable.initCache();

        matchToMatchLog10 = (double *)_mm_malloc(
            (((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1) * sizeof(double), ALIGN_SIZE);
        matchToMatchProb = (double *)_mm_malloc(
            (((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1) * sizeof(double), ALIGN_SIZE);
        qualToErrorProbCache =
            (double *)_mm_malloc((MAX_QUAL + 1) * sizeof(double), ALIGN_SIZE);
        qualToProbLog10Cache =
            (double *)_mm_malloc((MAX_QUAL + 1) * sizeof(double), ALIGN_SIZE);

        if (matchToMatchLog10 == NULL || matchToMatchProb == NULL ||
            qualToErrorProbCache == NULL || qualToProbLog10Cache == NULL)
        {
            free();
            return PDHMM_MEMORY_ALLOCATION_FAILED;
        }

        for (int32_t i = 0, offset = 0; i <= MAX_QUAL; offset += ++i)
        {
            for (int32_t j = 0; j <= i; j++)
            {
                double log10Sum = approximateLog10SumLog10(-0.1 * i, -0.1 * j);
                matchToMatchLog10[offset + j] =
                    log1p(-std::min(1.0, pow(10, log10Sum))) * INV_LN10;
                matchToMatchProb[offset + j] = pow(10, matchToMatchLog10[offset + j]);
            }
        }

        for (int32_t i = 0; i <= MAX_QUAL; i++)
        {
            qualToErrorProbCache[i] = qualToErrorProb((double)i, status);
            if (status != PDHMM_SUCCESS)
            {
                free();
                return status;
            }
            qualToProbLog10Cache[i] = log10(1.0 - qualToErrorProbCache[i]);
        }

        initialized = true;
        return status;
    }

    void free()
    {
        JacobianLogTable &jacobianLogTable = JacobianLogTable::getInstance();
        jacobianLogTable.freeCache();
        _mm_free(matchToMatchLog10);
        _mm_free(matchToMatchProb);
        _mm_free(qualToErrorProbCache);
        _mm_free(qualToProbLog10Cache);
        initialized = false;
    }

    double *getMatchToMatchLog10() const { return matchToMatchLog10; }
    double *getMatchToMatchProb() const { return matchToMatchProb; }
    double *getQualToErrorProbCache() const { return qualToErrorProbCache; }
    double *getQualToProbLog10Cache() const { return qualToProbLog10Cache; }

private:
    ProbabilityCache() : matchToMatchLog10(nullptr), matchToMatchProb(nullptr),
                         qualToErrorProbCache(nullptr), qualToProbLog10Cache(nullptr),
                         initialized(false) {}

    ~ProbabilityCache()
    {
        free();
    }

    ProbabilityCache(const ProbabilityCache &) = delete;
    ProbabilityCache &operator=(const ProbabilityCache &) = delete;

    double *matchToMatchLog10;
    double *matchToMatchProb;
    double *qualToErrorProbCache;
    double *qualToProbLog10Cache;
    bool initialized;
};

class DPTable
{
public:
    // Static method to get the single instance of the class
    static DPTable &getInstance()
    {
        static DPTable instance;
        return instance;
    }

    // Allocate or reallocate the tables based on the table sizes
    int32_t allocate(size_t dp_table_size, size_t transition_size, size_t prior_size)
    {
        int32_t status = PDHMM_SUCCESS;

        if (dp_table_size > this->dp_table_size || transition_size > this->transition_size || prior_size > this->prior_size)
        {
            free();

            this->dp_table_size = dp_table_size;
            matchMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);
            insertionMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);
            deletionMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);
            branchMatchMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);
            branchInsertionMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);
            branchDeletionMatrix = (double *)_mm_malloc(dp_table_size, ALIGN_SIZE);

            this->transition_size = transition_size;
            transition = (double *)_mm_malloc(transition_size, ALIGN_SIZE);

            this->prior_size = prior_size;
            prior = (double *)_mm_malloc(prior_size, ALIGN_SIZE);

            if (matchMatrix == nullptr || insertionMatrix == nullptr || deletionMatrix == nullptr || branchMatchMatrix == nullptr || branchInsertionMatrix == nullptr || branchDeletionMatrix == nullptr || transition == nullptr || prior == nullptr)
            {
                free();
                status = PDHMM_MEMORY_ALLOCATION_FAILED;
            }
        }
        return status;
    }

    // Free the tables
    void free()
    {
        _mm_free(matchMatrix);
        _mm_free(insertionMatrix);
        _mm_free(deletionMatrix);
        _mm_free(branchMatchMatrix);
        _mm_free(branchInsertionMatrix);
        _mm_free(branchDeletionMatrix);
        _mm_free(transition);
        _mm_free(prior);

        matchMatrix = nullptr;
        insertionMatrix = nullptr;
        deletionMatrix = nullptr;
        branchMatchMatrix = nullptr;
        branchInsertionMatrix = nullptr;
        branchDeletionMatrix = nullptr;
        transition = nullptr;
        prior = nullptr;

        dp_table_size = 0;
        transition_size = 0;
        prior_size = 0;
    }

    // Getter methods to access the tables
    double *getMatchMatrix() const { return matchMatrix; }
    double *getInsertionMatrix() const { return insertionMatrix; }
    double *getDeletionMatrix() const { return deletionMatrix; }
    double *getBranchMatchMatrix() const { return branchMatchMatrix; }
    double *getBranchInsertionMatrix() const { return branchInsertionMatrix; }
    double *getBranchDeletionMatrix() const { return branchDeletionMatrix; }
    double *getTransition() const { return transition; }
    double *getPrior() const { return prior; }

private:
    // Private constructor to prevent instantiation
    DPTable() : matchMatrix(nullptr), insertionMatrix(nullptr),
                deletionMatrix(nullptr), branchMatchMatrix(nullptr),
                branchInsertionMatrix(nullptr), branchDeletionMatrix(nullptr),
                transition(nullptr), prior(nullptr),
                dp_table_size(0), transition_size(0), prior_size(0) {}

    // Private destructor
    ~DPTable()
    {
        free();
    }

    // Delete copy constructor and assignment operator to prevent copying
    DPTable(const DPTable &) = delete;
    DPTable &operator=(const DPTable &) = delete;

    // Member variables to store the tables
    double *matchMatrix;
    double *insertionMatrix;
    double *deletionMatrix;
    double *branchMatchMatrix;
    double *branchInsertionMatrix;
    double *branchDeletionMatrix;
    double *transition;
    double *prior;
    size_t dp_table_size;
    size_t transition_size;
    size_t prior_size;
};

#endif
