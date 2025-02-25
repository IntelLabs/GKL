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
#include "IntelPDHMM.h"
#include <debug.h>
#include "pdhmm-serial.h"
#include "avx2_impl.h"
#include <omp.h>
#include "JavaData.h"
#ifndef __APPLE__
#include "avx512_impl.h"
#endif
#include <avx.h>
#ifdef __APPLE__
#include <cassert>
#endif

#include "pdhmm-implementation.h"
#include <chrono>

// int time_for_batch = 0;
// int time_for_compute = 0;

JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_initNative(JNIEnv *env, jclass obj, jclass readDataHolder, jclass haplotypeDataHolder, jint max_threads, jint avxLevel, jint maxMemoryInMB)
{
    initializeNative((int)max_threads, (AVXLevel)avxLevel, (int)maxMemoryInMB);
    try
    {
        JavaData::InitializeFieldIDs(env, readDataHolder, haplotypeDataHolder);
    }
    catch (JavaException &e)
    {
        jclass exceptionClass = env->FindClass(e.classPath);
        if (!exceptionClass)
        {
            env->FatalError("Unable to find Java exception class");
            return;
        }
        env->ThrowNew(exceptionClass, e.message);
    }
}

JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_computeLikelihoodsNative(JNIEnv *env, jobject obj, jobjectArray readDataArray, jobjectArray haplotypeDataArray, jdoubleArray likelihoodArray)
{
    // Step 0: Initialize JavaData with current data information
    ComputeConfig &config = ComputeConfig::getInstance();
    JavaData javaData(env, readDataArray, haplotypeDataArray, config.getMaxMemoryInMB());

    // Allocate DP Table based on max haplotype length
    allocateDPTable(javaData.getMaxHaplotypeLength(), javaData.getMaxReadLength());

    // Get the total number of batches
    int totalBatch = javaData.getTotalBatch();

    // Get the pointer to the likelihood array
    jdouble *likelihoods = env->GetDoubleArrayElements(likelihoodArray, NULL);
    if (likelihoods == NULL)
    {
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "Memory allocation issue.");
        return;
    }

    // Process each batch
    for (int i = 0; i < totalBatch; i++)
    {
        // auto start_time = std::chrono::high_resolution_clock::now();

        // Get the next batch of data
        PDHMMInputData currBatch = javaData.getNextBatch();

        // auto end_time = std::chrono::high_resolution_clock::now();
        // time_for_batch += static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

        // Compute PDHMM for the current batch
        // start_time = std::chrono::high_resolution_clock::now();

        int32_t status = computePDHMM(currBatch);
        // end_time = std::chrono::high_resolution_clock::now();
        // time_for_compute += static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

        if (status != PDHMM_SUCCESS)
        {
            // Release the likelihood array and throw an appropriate exception
            env->ReleaseDoubleArrayElements(likelihoodArray, likelihoods, 0);
            if (status == PDHMM_MEMORY_ALLOCATION_FAILED)
            {
                env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "Memory allocation issue.");
            }
            else if (status == PDHMM_INPUT_DATA_ERROR)
            {
                env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Error while calculating PDHMM. Input arrays aren't valid.");
            }
            else if (status == PDHMM_FAILURE)
            {
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Failure while computing PDHMM.");
            }
            else if (status == PDHMM_MEMORY_ACCESS_ERROR)
            {
                env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Out of bound memory access while computing PDHMM.");
            }
            return;
        }

        // Copy the results from the current batch to the likelihood array
        for (int j = 0; j < currBatch.getT(); j++)
        {
            likelihoods[i * javaData.getBatchSize() + j] = currBatch.getResult()[j];
        }
    }

    // Release the likelihood array
    env->ReleaseDoubleArrayElements(likelihoodArray, likelihoods, 0);
}

/*
 * Class:     com_intel_gkl_pdhmm_IntelPDHMM
 * Method:    computePDHMM
 * Signature: (Z)V
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_computePDHMMNative(JNIEnv *env, jobject obj, jbyteArray jhap_bases, jbyteArray jhap_pdbases, jbyteArray jread_bases, jbyteArray jread_qual, jbyteArray jread_ins_qual, jbyteArray jread_del_qual, jbyteArray jgcp, jlongArray jhap_lengths, jlongArray jread_lengths, jint testcase, jint maxHapLength, jint maxReadLength)
{
    // Allocate DP Table based on max haplotype length
    allocateDPTable(maxHapLength, maxReadLength);

    jdoubleArray jresult;
    jresult = env->NewDoubleArray(testcase);
    if (jresult == NULL)
    {
        env->ExceptionClear();
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "Memory allocation issue.");
        return NULL;
    }

    jbyte *hap_bases = (jbyte *)env->GetPrimitiveArrayCritical(jhap_bases, 0);
    jbyte *hap_pdbases = (jbyte *)env->GetPrimitiveArrayCritical(jhap_pdbases, 0);
    jbyte *read_bases = (jbyte *)env->GetPrimitiveArrayCritical(jread_bases, 0);
    jbyte *read_qual = (jbyte *)env->GetPrimitiveArrayCritical(jread_qual, 0);
    jbyte *read_ins_qual = (jbyte *)env->GetPrimitiveArrayCritical(jread_ins_qual, 0);
    jbyte *read_del_qual = (jbyte *)env->GetPrimitiveArrayCritical(jread_del_qual, 0);
    jbyte *gcp = (jbyte *)env->GetPrimitiveArrayCritical(jgcp, 0);
    jlong *hap_lengths = (jlong *)env->GetPrimitiveArrayCritical(jhap_lengths, 0);
    jlong *read_lengths = (jlong *)env->GetPrimitiveArrayCritical(jread_lengths, 0);

    if (hap_bases == NULL || hap_pdbases == NULL || read_bases == NULL || read_qual == NULL || read_ins_qual == NULL || read_del_qual == NULL || gcp == NULL || hap_lengths == NULL || read_lengths == NULL)
    {
        DBG("GetPrimitiveArrayCritical failed from JAVA unable to continue.");
        if (env->ExceptionCheck())
            env->ExceptionClear();
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Input arrays aren't valid.");
        if (hap_bases != NULL)
            env->ReleasePrimitiveArrayCritical(jhap_bases, hap_bases, 0);
        if (hap_pdbases != NULL)
            env->ReleasePrimitiveArrayCritical(jhap_pdbases, hap_pdbases, 0);
        if (read_bases != NULL)
            env->ReleasePrimitiveArrayCritical(jread_bases, read_bases, 0);
        if (read_qual != NULL)
            env->ReleasePrimitiveArrayCritical(jread_qual, read_qual, 0);
        if (read_ins_qual != NULL)
            env->ReleasePrimitiveArrayCritical(jread_ins_qual, read_ins_qual, 0);
        if (read_del_qual != NULL)
            env->ReleasePrimitiveArrayCritical(jread_del_qual, read_del_qual, 0);
        if (gcp != NULL)
            env->ReleasePrimitiveArrayCritical(jgcp, gcp, 0);
        if (hap_lengths != NULL)
            env->ReleasePrimitiveArrayCritical(jhap_lengths, hap_lengths, 0);
        if (read_lengths != NULL)
            env->ReleasePrimitiveArrayCritical(jread_lengths, read_lengths, 0);
        return NULL;
    }

    double *result = (double *)_mm_malloc(testcase * sizeof(double), ALIGN_SIZE);
    if (result == NULL)
    {
        env->ExceptionClear();
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "Memory allocation issue.");
        return NULL;
    }
    int32_t status = computePDHMM(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, testcase, (int64_t *)hap_lengths, (int64_t *)read_lengths, maxReadLength, maxHapLength);

    // release buffers
    env->ReleasePrimitiveArrayCritical(jhap_bases, hap_bases, 0);
    env->ReleasePrimitiveArrayCritical(jhap_pdbases, hap_pdbases, 0);
    env->ReleasePrimitiveArrayCritical(jread_bases, read_bases, 0);
    env->ReleasePrimitiveArrayCritical(jread_qual, read_qual, 0);
    env->ReleasePrimitiveArrayCritical(jread_ins_qual, read_ins_qual, 0);
    env->ReleasePrimitiveArrayCritical(jread_del_qual, read_del_qual, 0);
    env->ReleasePrimitiveArrayCritical(jgcp, gcp, 0);
    env->ReleasePrimitiveArrayCritical(jhap_lengths, hap_lengths, 0);
    env->ReleasePrimitiveArrayCritical(jread_lengths, read_lengths, 0);

    if (status != PDHMM_SUCCESS)
    {
        if (status == PDHMM_MEMORY_ALLOCATION_FAILED)
        {
            env->ExceptionClear();
            env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), "Memory allocation issue.");
        }
        if (status == PDHMM_INPUT_DATA_ERROR)
        {
            env->ExceptionClear();
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Error while calculating pdhmm. Input arrays aren't valid.");
        }
        if (status == PDHMM_FAILURE)
        {
            env->ExceptionClear();
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Failure while computing PDHMM.");
        }
        if (status == PDHMM_MEMORY_ACCESS_ERROR)
        {
            env->ExceptionClear();
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Out of bound memory access while computing PDHMM.");
        }
    }
    else
    {
        env->SetDoubleArrayRegion(jresult, 0, testcase, result);
    }
    _mm_free(result);
    return jresult;
}

JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_doneNative(JNIEnv *env, jclass obj)
{
    doneNative();
    // INFO("Total time for creating batch: %d ms", time_for_batch);
    // INFO("Total time for computing pdhmm: %d ms", time_for_compute);
}
