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
package com.intel.gkl.pdhmm;

import com.intel.gkl.IntelGKLUtils;
import com.intel.gkl.NativeLibraryLoader;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.broadinstitute.gatk.nativebindings.pdhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pdhmm.ReadDataHolder;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeArguments.AVXLevel;

import java.io.File;
import java.lang.reflect.Array;
import java.util.Objects;

/**
 * Provides a native PDHMM implementation accelerated for the Intel
 * Architecture.
 */
public class IntelPDHMM implements PDHMMNativeBinding {
    private final static Log logger = LogFactory.getLog(IntelPDHMM.class);
    private static final String NATIVE_LIBRARY_NAME = "gkl_pdhmm";
    private IntelGKLUtils gklUtils = new IntelGKLUtils();

    /**
     * Loads the Intel GKL Utils and the native library.
     * This method is synchronized to ensure thread safety.
     *
     * @param tempDir the temporary directory where the native library is located
     * @return true if both the Intel GKL Utils and the native library are
     *         successfully loaded, false otherwise
     */
    @Override
    public synchronized boolean load(File tempDir) {
        boolean isLoaded = gklUtils.load(null);

        if (!isLoaded) {
            logger.warn("Intel GKL Utils not loaded");
            return false;
        }

        return NativeLibraryLoader.load(tempDir, NATIVE_LIBRARY_NAME);

    }

    /**
     * Initializes the IntelPDHMM instance with the provided native arguments.
     *
     * @param args the PDHMMNativeArguments containing the initialization
     *             parameters.
     */
    @Override
    public void initialize(PDHMMNativeArguments args) {
        if (args == null) {
            args = new PDHMMNativeArguments();
            args.maxNumberOfThreads = 1;
            args.avxLevel = AVXLevel.AVX512;
            args.maxMemoryInMB = 512;
        }
        initNative(ReadDataHolder.class, HaplotypeDataHolder.class,
                args.maxNumberOfThreads, args.avxLevel.ordinal(), args.maxMemoryInMB);
    }

    @Override
    public void computeLikelihoods(ReadDataHolder[] readDataArray, HaplotypeDataHolder[] haplotypeDataArray,
            double[] likelihoodArray) throws NullPointerException, OutOfMemoryError, IllegalArgumentException {
        if (readDataArray == null || haplotypeDataArray == null || likelihoodArray == null) {
            throw new NullPointerException(
                    "One or more input arrays are null. Please ensure readDataArray, haplotypeDataArray, and likelihoodArray are properly initialized.");
        }

        if (likelihoodArray.length != readDataArray.length * haplotypeDataArray.length) {
            throw new IllegalArgumentException(
                    "likelihoodArray length must be equal to readDataArray length * haplotypeDataArray length");
        }

        try {
            computeLikelihoodsNative(readDataArray, haplotypeDataArray, likelihoodArray);
        } catch (OutOfMemoryError e) {
            logger.warn(
                    "Exception thrown from native PDHMM computeLikelihoodsNative function call " + e.getMessage());
            throw new OutOfMemoryError("Memory allocation failed");
        } catch (IllegalArgumentException e) {
            logger.warn(
                    "Exception thrown from native PDHMM computeLikelihoodsNative function call " + e.getMessage());
            throw new IllegalArgumentException("Ran into invalid argument issue");
        } catch (RuntimeException e) {
            logger.warn(
                    "Exception thrown from native PDHMM computeLikelihoodsNative function call " + e.getMessage());
            throw new RuntimeException(
                    "Runtime exception thrown from native pdhmm function call " + e.getMessage());
        }

    }

    @Override
    public void done() {
        doneNative();
    }

    private static void checkArraySize(Object array, int expectedSize, String arrayName) {
        Objects.requireNonNull(array, arrayName + " must not be null.");
        if (!array.getClass().isArray()) {
            throw new IllegalArgumentException(arrayName + " is not an array.");
        } else if (Array.getLength(array) != expectedSize) {
            String errorMessage = String.format("Array %s has size %d, but expected size is %d.", arrayName,
                    Array.getLength(
                            array),
                    expectedSize);
            throw new IllegalArgumentException(errorMessage);
        }
    }

    /**
     * Computes the PDHMM for the given batch of input data.
     *
     * @param hap_bases     the haplotype bases
     * @param hap_pdbases   the haplotype PD bases
     * @param read_bases    the read bases
     * @param read_qual     the read quality scores
     * @param read_ins_qual the read insertion quality scores
     * @param read_del_qual the read deletion quality scores
     * @param gcp           the gap compression penalties
     * @param hap_lengths   the lengths of the haplotypes
     * @param read_lengths  the lengths of the reads
     * @param batchSize     the number of batches
     * @param maxHapLength  the maximum length of the haplotypes
     * @param maxReadLength the maximum length of the reads
     * @return an array of computed PDHMM values
     * @throws IllegalArgumentException if any of the input arrays are null or have
     *                                  incorrect sizes
     * @throws OutOfMemoryError         if memory allocation fails
     * @throws RuntimeException         if a runtime exception occurs during the
     *                                  native function call
     */
    public double[] computePDHMM(byte[] hap_bases, byte[] hap_pdbases, byte[] read_bases, byte[] read_qual,
            byte[] read_ins_qual, byte[] read_del_qual, byte[] gcp, long[] hap_lengths, long[] read_lengths,
            int batchSize, int maxHapLength, int maxReadLength) {
        int hapArrayLength = maxHapLength * batchSize;
        int readArrayLength = maxReadLength * batchSize;

        checkArraySize(hap_bases, hapArrayLength, "hap_bases");
        checkArraySize(hap_pdbases, hapArrayLength, "hap_pdbases");
        checkArraySize(read_bases, readArrayLength, "read_bases");
        checkArraySize(read_qual, readArrayLength, "read_qual");
        checkArraySize(read_ins_qual, readArrayLength, "read_ins_qual");
        checkArraySize(read_del_qual, readArrayLength, "read_del_qual");
        checkArraySize(gcp, readArrayLength, "gcp");
        checkArraySize(hap_lengths, batchSize, "hap_lengths");
        checkArraySize(read_lengths, batchSize, "read_lengths");

        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize must be greater than 0.");

        if (maxHapLength <= 0)
            throw new IllegalArgumentException(
                    "maxHapLength must be greater than 0. Cannot perform PDHMM on empty sequence");

        if (maxReadLength <= 0)
            throw new IllegalArgumentException(
                    "maxReadLength must be greater than 0. Cannot perform PDHMM on empty sequence");

        try {
            return computePDHMMNative(hap_bases, hap_pdbases, read_bases, read_qual,
                    read_ins_qual, read_del_qual, gcp, hap_lengths, read_lengths,
                    batchSize, maxHapLength, maxReadLength);
        } catch (OutOfMemoryError e) {
            throw new OutOfMemoryError(
                    "OutOfMemory exception thrown from native pdhmm function call " + e.getMessage());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "IllegalArgument exception thrown from native pdhmm function call " + e.getMessage());
        } catch (RuntimeException e) {
            throw new RuntimeException(
                    "Runtime exception thrown from native pdhmm function call " + e.getMessage());
        }
    }

    private native static void initNative(Class<?> readDataHolderClass,
            Class<?> haplotypeDataHolderClass, int maxThreads, int avxLevel, int maxMemoryInMB);

    private native double[] computePDHMMNative(byte[] hap_bases, byte[] hap_pdbases, byte[] read_bases,
            byte[] read_qual,
            byte[] read_ins_qual, byte[] read_del_qual, byte[] gcp, long[] hap_lengths, long[] read_lengths,
            int testcase, int maxHapLength, int maxReadLength);

    private native void computeLikelihoodsNative(Object[] readDataArray,
            Object[] haplotypeDataArray,
            double[] likelihoodArray);

    private native static void doneNative();

}
