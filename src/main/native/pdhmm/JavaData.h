#ifndef JAVADATA_H
#define JAVADATA_H

#include <vector>
#include <exception>
#include <string>
#include <algorithm>
#include "pdhmm-common.h"

#include <iostream>
#include <cmath>
#include <immintrin.h>

class JavaException : public std::exception
{
public:
    const char *classPath;
    const char *message;

    JavaException(const char *classPath, const char *message)
        : classPath(classPath),
          message(message)
    {
    }
};

class JavaData
{
public:
    // Constructor to initialize the values specific to the input data
    JavaData(JNIEnv *env, jobjectArray &readDataArray, jobjectArray &haplotypeDataArray, int maxMemoryInMB)
        : env(env),
          readDataArray(readDataArray),
          haplotypeDataArray(haplotypeDataArray),
          totalReads(env->GetArrayLength(readDataArray)),
          totalHaplotypes(env->GetArrayLength(haplotypeDataArray)),
          totalPairs(totalReads * totalHaplotypes),
          maxReadLength(0),
          maxHaplotypeLength(0),
          batchSize(0),
          totalBatch(0),
          currentBatchIndex(0)
    {
        if (totalReads == 0 || totalHaplotypes == 0)
        {
            throw JavaException("java/lang/IllegalArgumentException", "Input arrays are empty.");
        }

        // Calculate maxReadLength
        for (int i = 0; i < totalReads; i++)
        {
            jobject readData = env->GetObjectArrayElement(readDataArray, i);
            jbyteArray readBases = (jbyteArray)env->GetObjectField(readData, m_readBasesFid);
            int readLength = env->GetArrayLength(readBases);
            maxReadLength = std::max(maxReadLength, readLength);
            env->DeleteLocalRef(readBases);
            env->DeleteLocalRef(readData);
        }

        // Calculate maxHaplotypeLength
        for (int i = 0; i < totalHaplotypes; i++)
        {
            jobject haplotypeData = env->GetObjectArrayElement(haplotypeDataArray, i);
            jbyteArray haplotypeBases = (jbyteArray)env->GetObjectField(haplotypeData, m_haplotypeBasesFid);
            int haplotypeLength = env->GetArrayLength(haplotypeBases);
            maxHaplotypeLength = std::max(maxHaplotypeLength, haplotypeLength);
            env->DeleteLocalRef(haplotypeBases);
            env->DeleteLocalRef(haplotypeData);
        }

        // Compute Batch information

        int64_t maxMemory = static_cast<int64_t>(maxMemoryInMB) * 1024 * 1024; // Convert to bytes
        int64_t memoryPerPair = (maxReadLength * 5 + maxHaplotypeLength * 2) * sizeof(int8_t) + sizeof(double) + 2 * sizeof(int64_t);
        batchSize = std::min(totalPairs, static_cast<int>(maxMemory / memoryPerPair));
        totalBatch = static_cast<int>(std::ceil(static_cast<double>(totalPairs) / batchSize));

        // Allocate memory for the entire batch
        allocateBatchMemory();

        // Read Java arrays once and store them in native arrays
        readJavaArrays();
    }

    // Destructor
    virtual ~JavaData()
    {
        freeBatchMemory();

        // Release the memory allocated for the native arrays
        for (int i = 0; i < totalReads; i++)
        {
            delete[] nativeReadBases[i];
            delete[] nativeReadQuals[i];
            delete[] nativeInsertionGop[i];
            delete[] nativeDeletionGop[i];
            delete[] nativeOverallGcp[i];
        }

        for (int i = 0; i < totalHaplotypes; i++)
        {
            delete[] nativeHapBases[i];
            delete[] nativeHapPDBases[i];
        }

        // Clear the vectors
        nativeReadBases.clear();
        nativeReadQuals.clear();
        nativeInsertionGop.clear();
        nativeDeletionGop.clear();
        nativeOverallGcp.clear();
        nativeReadLengths.clear();
        nativeHapBases.clear();
        nativeHapPDBases.clear();
        nativeHapLengths.clear();
    }

    // Delete copy constructor and assignment operator to prevent copying
    JavaData(const JavaData &) = delete;
    JavaData &operator=(const JavaData &) = delete;

    // Print data
    void printData() const
    {
        std::cout << "Total Reads: " << totalReads << std::endl;
        std::cout << "Total Haplotypes: " << totalHaplotypes << std::endl;
        std::cout << "Total Pairs: " << totalPairs << std::endl;
        std::cout << "Max Read Length: " << maxReadLength << std::endl;
        std::cout << "Max Haplotype Length: " << maxHaplotypeLength << std::endl;
        std::cout << "Batch Size: " << batchSize << std::endl;
        std::cout << "Total Batch: " << totalBatch << std::endl;
    }

    // Getter methods
    int getTotalReads() const { return totalReads; }
    int getTotalHaplotypes() const { return totalHaplotypes; }
    int getTotalPairs() const { return totalPairs; }
    int getMaxReadLength() const { return maxReadLength; }
    int getMaxHaplotypeLength() const { return maxHaplotypeLength; }
    int getBatchSize() const { return batchSize; }
    int getTotalBatch() const { return totalBatch; }

    // Initializes the static field IDs for the readDataHolder and haplotypeDataHolder classes
    static void InitializeFieldIDs(JNIEnv *env, jclass readDataHolder, jclass haplotypeDataHolder)
    {
        m_readBasesFid = getFieldId(env, readDataHolder, "readBases", "[B");
        m_readQualsFid = getFieldId(env, readDataHolder, "readQuals", "[B");
        m_insertionGopFid = getFieldId(env, readDataHolder, "insertionGOP", "[B");
        m_deletionGopFid = getFieldId(env, readDataHolder, "deletionGOP", "[B");
        m_overallGcpFid = getFieldId(env, readDataHolder, "overallGCP", "[B");
        m_haplotypeBasesFid = getFieldId(env, haplotypeDataHolder, "haplotypeBases", "[B");
        m_haplotypePDBasesFid = getFieldId(env, haplotypeDataHolder, "haplotypePDBases", "[B");
    }

    // Function to get the next batch of data
    PDHMMInputData getNextBatch()
    {
        if (currentBatchIndex >= totalBatch)
        {
            throw JavaException("java/lang/IndexOutOfBoundsException", "No more batches available.");
        }

        int startPairIndex = currentBatchIndex * batchSize;
        int endPairIndex = std::min(startPairIndex + batchSize, totalPairs);
        int currBatchSize = endPairIndex - startPairIndex;
        int currPairId = 0;
        int pairIndex = startPairIndex;
        int currReadIndex = pairIndex / totalHaplotypes;
        int currHaplotypeIndex = pairIndex % totalHaplotypes;

        for (; currReadIndex < totalReads && currPairId < currBatchSize; ++currReadIndex)
        {
            int8_t *readBases = nativeReadBases[currReadIndex];
            int8_t *readQuals = nativeReadQuals[currReadIndex];
            int8_t *insertionGop = nativeInsertionGop[currReadIndex];
            int8_t *deletionGop = nativeDeletionGop[currReadIndex];
            int8_t *overallGcp = nativeOverallGcp[currReadIndex];
            int readLength = nativeReadLengths[currReadIndex];

            for (; currHaplotypeIndex < totalHaplotypes && currPairId < currBatchSize; ++currHaplotypeIndex, ++currPairId)
            {
                int8_t *hapBases = nativeHapBases[currHaplotypeIndex];
                int8_t *hapPDBases = nativeHapPDBases[currHaplotypeIndex];
                int haplotypeLength = nativeHapLengths[currHaplotypeIndex];

                // Copy data to the allocated batch memory
                std::copy(readBases, readBases + readLength, batchReadBases + currPairId * maxReadLength);
                std::fill(batchReadBases + currPairId * maxReadLength + readLength, batchReadBases + (currPairId + 1) * maxReadLength, 0); // Padding

                std::copy(readQuals, readQuals + readLength, batchReadQuals + currPairId * maxReadLength);
                std::fill(batchReadQuals + currPairId * maxReadLength + readLength, batchReadQuals + (currPairId + 1) * maxReadLength, 0); // Padding

                std::copy(insertionGop, insertionGop + readLength, batchInsertionGop + currPairId * maxReadLength);
                std::fill(batchInsertionGop + currPairId * maxReadLength + readLength, batchInsertionGop + (currPairId + 1) * maxReadLength, 0); // Padding

                std::copy(deletionGop, deletionGop + readLength, batchDeletionGop + currPairId * maxReadLength);
                std::fill(batchDeletionGop + currPairId * maxReadLength + readLength, batchDeletionGop + (currPairId + 1) * maxReadLength, 0); // Padding

                std::copy(overallGcp, overallGcp + readLength, batchOverallGcp + currPairId * maxReadLength);
                std::fill(batchOverallGcp + currPairId * maxReadLength + readLength, batchOverallGcp + (currPairId + 1) * maxReadLength, 0); // Padding

                std::copy(hapBases, hapBases + haplotypeLength, batchHapBases + currPairId * maxHaplotypeLength);
                std::fill(batchHapBases + currPairId * maxHaplotypeLength + haplotypeLength, batchHapBases + (currPairId + 1) * maxHaplotypeLength, 0); // Padding

                std::copy(hapPDBases, hapPDBases + haplotypeLength, batchHapPDBases + currPairId * maxHaplotypeLength);
                std::fill(batchHapPDBases + currPairId * maxHaplotypeLength + haplotypeLength, batchHapPDBases + (currPairId + 1) * maxHaplotypeLength, 0); // Padding

                batchHapLengths[currPairId] = haplotypeLength;
                batchReadLengths[currPairId] = readLength;
                batchResult[currPairId] = 0.0; // Initialize result to 0.0
            }
            currHaplotypeIndex = 0; // Reset haplotype index for the next read
        }

        currentBatchIndex++;

        return PDHMMInputData(
            batchHapBases, batchHapPDBases, batchReadBases, batchReadQuals,
            batchInsertionGop, batchDeletionGop, batchOverallGcp, batchResult,
            currBatchSize, batchHapLengths, batchReadLengths, maxReadLength, maxHaplotypeLength);
    }

private:
    // Member variables
    JNIEnv *env;
    jobjectArray readDataArray;
    jobjectArray haplotypeDataArray;
    int totalReads;
    int totalHaplotypes;
    int totalPairs;
    int maxReadLength;
    int maxHaplotypeLength;
    int batchSize;
    int totalBatch;
    int currentBatchIndex;

    // Batch memory
    int8_t *batchReadBases;
    int8_t *batchReadQuals;
    int8_t *batchInsertionGop;
    int8_t *batchDeletionGop;
    int8_t *batchOverallGcp;
    int8_t *batchHapBases;
    int8_t *batchHapPDBases;
    double *batchResult;
    int64_t *batchHapLengths;
    int64_t *batchReadLengths;

    // Native arrays to store Java array data
    std::vector<int8_t *> nativeReadBases;
    std::vector<int8_t *> nativeReadQuals;
    std::vector<int8_t *> nativeInsertionGop;
    std::vector<int8_t *> nativeDeletionGop;
    std::vector<int8_t *> nativeOverallGcp;
    std::vector<int> nativeReadLengths;
    std::vector<int8_t *> nativeHapBases;
    std::vector<int8_t *> nativeHapPDBases;
    std::vector<int> nativeHapLengths;

    // Helper function to get field ID
    static jfieldID getFieldId(JNIEnv *env, jclass clazz, const char *name, const char *sig)
    {
        jfieldID id = env->GetFieldID(clazz, name, sig);
        if (id == NULL)
        {
            throw JavaException("java/lang/IllegalArgumentException", "Unable to get field ID");
        }
        return id;
    }

    // Static field IDs
    static jfieldID m_readBasesFid;
    static jfieldID m_readQualsFid;
    static jfieldID m_insertionGopFid;
    static jfieldID m_deletionGopFid;
    static jfieldID m_overallGcpFid;
    static jfieldID m_haplotypeBasesFid;
    static jfieldID m_haplotypePDBasesFid;

    // Allocate memory for the entire batch
    void allocateBatchMemory()
    {
        batchReadBases = (int8_t *)_mm_malloc(batchSize * maxReadLength * sizeof(int8_t), ALIGN_SIZE);
        batchReadQuals = (int8_t *)_mm_malloc(batchSize * maxReadLength * sizeof(int8_t), ALIGN_SIZE);
        batchInsertionGop = (int8_t *)_mm_malloc(batchSize * maxReadLength * sizeof(int8_t), ALIGN_SIZE);
        batchDeletionGop = (int8_t *)_mm_malloc(batchSize * maxReadLength * sizeof(int8_t), ALIGN_SIZE);
        batchOverallGcp = (int8_t *)_mm_malloc(batchSize * maxReadLength * sizeof(int8_t), ALIGN_SIZE);
        batchHapBases = (int8_t *)_mm_malloc(batchSize * maxHaplotypeLength * sizeof(int8_t), ALIGN_SIZE);
        batchHapPDBases = (int8_t *)_mm_malloc(batchSize * maxHaplotypeLength * sizeof(int8_t), ALIGN_SIZE);
        batchResult = (double *)_mm_malloc(batchSize * sizeof(double), ALIGN_SIZE);
        batchHapLengths = (int64_t *)_mm_malloc(batchSize * sizeof(int64_t), ALIGN_SIZE);
        batchReadLengths = (int64_t *)_mm_malloc(batchSize * sizeof(int64_t), ALIGN_SIZE);
    }

    // Free the allocated batch memory
    void freeBatchMemory()
    {
        _mm_free(batchReadBases);
        _mm_free(batchReadQuals);
        _mm_free(batchInsertionGop);
        _mm_free(batchDeletionGop);
        _mm_free(batchOverallGcp);
        _mm_free(batchHapBases);
        _mm_free(batchHapPDBases);
        _mm_free(batchResult);
        _mm_free(batchHapLengths);
        _mm_free(batchReadLengths);
    }

    // Read Java arrays once and store them in native arrays
    void readJavaArrays()
    {
        nativeReadBases.resize(totalReads);
        nativeReadQuals.resize(totalReads);
        nativeInsertionGop.resize(totalReads);
        nativeDeletionGop.resize(totalReads);
        nativeOverallGcp.resize(totalReads);
        nativeReadLengths.resize(totalReads);

        for (int i = 0; i < totalReads; i++)
        {
            jobject readData = env->GetObjectArrayElement(readDataArray, i);
            jbyteArray readBasesArray = (jbyteArray)env->GetObjectField(readData, m_readBasesFid);
            jbyteArray readQualsArray = (jbyteArray)env->GetObjectField(readData, m_readQualsFid);
            jbyteArray insertionGopArray = (jbyteArray)env->GetObjectField(readData, m_insertionGopFid);
            jbyteArray deletionGopArray = (jbyteArray)env->GetObjectField(readData, m_deletionGopFid);
            jbyteArray overallGcpArray = (jbyteArray)env->GetObjectField(readData, m_overallGcpFid);

            int readBasesLength = env->GetArrayLength(readBasesArray);
            int readQualsLength = env->GetArrayLength(readQualsArray);
            int insertionGopLength = env->GetArrayLength(insertionGopArray);
            int deletionGopLength = env->GetArrayLength(deletionGopArray);
            int overallGcpLength = env->GetArrayLength(overallGcpArray);

            nativeReadBases[i] = new int8_t[readBasesLength];
            nativeReadQuals[i] = new int8_t[readQualsLength];
            nativeInsertionGop[i] = new int8_t[insertionGopLength];
            nativeDeletionGop[i] = new int8_t[deletionGopLength];
            nativeOverallGcp[i] = new int8_t[overallGcpLength];

            env->GetByteArrayRegion(readBasesArray, 0, readBasesLength, nativeReadBases[i]);
            env->GetByteArrayRegion(readQualsArray, 0, readQualsLength, nativeReadQuals[i]);
            env->GetByteArrayRegion(insertionGopArray, 0, insertionGopLength, nativeInsertionGop[i]);
            env->GetByteArrayRegion(deletionGopArray, 0, deletionGopLength, nativeDeletionGop[i]);
            env->GetByteArrayRegion(overallGcpArray, 0, overallGcpLength, nativeOverallGcp[i]);

            nativeReadLengths[i] = readBasesLength;

            env->DeleteLocalRef(readBasesArray);
            env->DeleteLocalRef(readQualsArray);
            env->DeleteLocalRef(insertionGopArray);
            env->DeleteLocalRef(deletionGopArray);
            env->DeleteLocalRef(overallGcpArray);
            env->DeleteLocalRef(readData);
        }

        nativeHapBases.resize(totalHaplotypes);
        nativeHapPDBases.resize(totalHaplotypes);
        nativeHapLengths.resize(totalHaplotypes);

        for (int i = 0; i < totalHaplotypes; i++)
        {
            jobject haplotypeData = env->GetObjectArrayElement(haplotypeDataArray, i);
            jbyteArray haplotypeBasesArray = (jbyteArray)env->GetObjectField(haplotypeData, m_haplotypeBasesFid);
            jbyteArray haplotypePDBasesArray = (jbyteArray)env->GetObjectField(haplotypeData, m_haplotypePDBasesFid);

            int haplotypeBasesLength = env->GetArrayLength(haplotypeBasesArray);
            int haplotypePDBasesLength = env->GetArrayLength(haplotypePDBasesArray);

            nativeHapBases[i] = new int8_t[haplotypeBasesLength];
            nativeHapPDBases[i] = new int8_t[haplotypePDBasesLength];

            env->GetByteArrayRegion(haplotypeBasesArray, 0, haplotypeBasesLength, nativeHapBases[i]);
            env->GetByteArrayRegion(haplotypePDBasesArray, 0, haplotypePDBasesLength, nativeHapPDBases[i]);

            nativeHapLengths[i] = haplotypeBasesLength;

            env->DeleteLocalRef(haplotypeBasesArray);
            env->DeleteLocalRef(haplotypePDBasesArray);
            env->DeleteLocalRef(haplotypeData);
        }
    }
};

// Definition of static field IDs
jfieldID JavaData::m_readBasesFid;
jfieldID JavaData::m_readQualsFid;
jfieldID JavaData::m_insertionGopFid;
jfieldID JavaData::m_deletionGopFid;
jfieldID JavaData::m_overallGcpFid;
jfieldID JavaData::m_haplotypeBasesFid;
jfieldID JavaData::m_haplotypePDBasesFid;

#endif // JAVADATA_H
