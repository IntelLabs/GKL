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
#ifndef _Included_com_intel_gkl_pdhmm_IntelPDHMM
#define _Included_com_intel_gkl_pdhmm_IntelPDHMM
#include <jni.h>
#ifdef __cplusplus
extern "C"
{
#endif

    JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_initNative(JNIEnv *env, jclass obj, jclass readDataHolder, jclass haplotypeDataHolder, jint max_threads, jint avxLevel, jint maxMemoryInMB);

    /*
     * Class:     com_intel_gkl_pdhmm_IntelPDHMM
     * Method:    computeReadLikelihoodGivenHaplotypeLog10Vec
     * Signature: ()Z
     */
    JNIEXPORT jdoubleArray JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_computePDHMMNative(JNIEnv *env, jobject obj, jbyteArray jhap_bases, jbyteArray jhap_pdbases, jbyteArray jread_bases, jbyteArray jread_qual, jbyteArray jread_ins_qual, jbyteArray jread_del_qual, jbyteArray jgcp, jlongArray jhap_lengths, jlongArray jread_lengths, jint testcase, jint maxHapLength, jint maxReadLength);

    /*
     * Class:     com_intel_gkl_pairhmm_IntelPairHmm
     * Method:    computeLikelihoodsNative
     * Signature: ([Ljava/lang/Object;[Ljava/lang/Object;[D)V
     */
    JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_computeLikelihoodsNative(JNIEnv *, jobject, jobjectArray, jobjectArray, jdoubleArray);

    JNIEXPORT void JNICALL Java_com_intel_gkl_pdhmm_IntelPDHMM_doneNative(JNIEnv *env, jclass obj);

#ifdef __cplusplus
}
#endif
#endif
