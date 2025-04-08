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
#ifndef PDHMM_IMPLEMENTATION_H
#define PDHMM_IMPLEMENTATION_H

#include "pdhmm-common.h"
#include "pdhmm-serial.h"
#include "avx2_impl.h"
#ifndef __APPLE__
#include "avx512_impl.h"
#endif

#if defined(__linux__)
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

enum class AVXLevel
{
    FASTEST_AVAILABLE,
    SCALAR,
    AVX2,
    AVX512
};

enum class OpenMPSetting
{
    FASTEST_AVAILABLE,
    ENABLE,
    DISABLE
};

class ComputeConfig
{
public:
    // Static method to get the single instance of the class
    static ComputeConfig &getInstance()
    {
        static ComputeConfig instance;
        return instance;
    }

    // Initialize the configuration based on system capabilities and user requirements
    void initialize(OpenMPSetting openMPSetting = OpenMPSetting::FASTEST_AVAILABLE, int numThreads = 1, AVXLevel userAVXLevel = AVXLevel::AVX512, int maxMemoryInMB = 512)
    {
        // Check OpenMP setting
        switch (static_cast<int>(openMPSetting))
        {
        case static_cast<int>(OpenMPSetting::FASTEST_AVAILABLE):
            this->openMP = numThreads > 1 && isOpenMPAvailable();

            break;
        case static_cast<int>(OpenMPSetting::ENABLE):
            if (!isOpenMPAvailable())
            {
                throw JavaException("java/lang/IllegalStateException", "OpenMP is enabled but not available on this system. Please disable OpenMP or run on a system that supports OpenMP.");
            }
            this->openMP = true;
            break;
        case static_cast<int>(OpenMPSetting::DISABLE):
        default:
            this->openMP = false;
            numThreads = 1;
            break;
        }

        // Check AVX level setting
        switch (static_cast<int>(userAVXLevel))
        {
        case static_cast<int>(AVXLevel::FASTEST_AVAILABLE):
            this->avxLevel = getBestAvailableAVXLevel(userAVXLevel);
            break;
        case static_cast<int>(AVXLevel::AVX512):
        case static_cast<int>(AVXLevel::AVX2):
        case static_cast<int>(AVXLevel::SCALAR):
            if (!isArchSupported(userAVXLevel))
            {
                throw JavaException("java/lang/IllegalStateException", "Requested AVX level is not available on this system.");
            }
            this->avxLevel = userAVXLevel;
            break;
        default:
            this->avxLevel = detectBestAVXLevel();
            break;
        }

        this->numThreads = this->openMP ? getBestAvailableNumThreads(numThreads) : 1;
        this->maxMemoryInMB = getMaxMemoryAvailable(maxMemoryInMB);
        printConfig();
    }

    // Getter methods
    bool isOpenMPEnabled() const { return openMP; }
    AVXLevel getAVXLevel() const { return avxLevel; }
    int getNumThreads() const { return numThreads; }
    int getMaxMemoryInMB() const { return maxMemoryInMB; }

private:
    // Private constructor to prevent instantiation
    ComputeConfig() : openMP(false), avxLevel(AVXLevel::SCALAR), numThreads(1), maxMemoryInMB(512) {}

    // Private destructor
    ~ComputeConfig() {}

    // Delete copy constructor and assignment operator to prevent copying
    ComputeConfig(const ComputeConfig &) = delete;
    ComputeConfig &operator=(const ComputeConfig &) = delete;

    // Check if the specified AVX level is supported
    bool isArchSupported(AVXLevel avxLevel)
    {
        switch (static_cast<int>(avxLevel))
        {
        case static_cast<int>(AVXLevel::AVX512):
            return is_avx512_supported();
        case static_cast<int>(AVXLevel::AVX2):
            return is_avx_supported() && is_avx2_supported() && is_sse_supported();
        case static_cast<int>(AVXLevel::SCALAR):
            return true;
        default:
            return false;
        }
    }

    // Detect the system's best available AVX capabilities
    AVXLevel detectBestAVXLevel()
    {
        if (isArchSupported(AVXLevel::AVX512))
        {
            return AVXLevel::AVX512;
        }
        else if (isArchSupported(AVXLevel::AVX2))
        {
            return AVXLevel::AVX2;
        }
        else
        {
            return AVXLevel::SCALAR;
        }
    }

    // Get the best available AVX level based on user preference
    AVXLevel getBestAvailableAVXLevel(AVXLevel userAVXLevel)
    {
        if (isArchSupported(userAVXLevel))
        {
            return userAVXLevel;
        }
        else
        {
            return detectBestAVXLevel();
        }
    }

    // Check if OpenMP is available
    bool isOpenMPAvailable()
    {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }

    // Get the best available number of threads based on user preference
    int getBestAvailableNumThreads(int numThreads)
    {
#ifdef _OPENMP
        int availThreads = omp_get_max_threads();
        return std::min(numThreads, availThreads);
#else
        return 1;
#endif
    }

    // Get the maximum memory available on the system
    int getMaxMemoryAvailable(int maxMemoryInMB)
    {
        if (maxMemoryInMB <= 0)
        {
            throw JavaException("java/lang/IllegalArgumentException", "Max memory should be greater than 0.");
        }

        int systemMaxMemoryInMB = maxMemoryInMB;
#if defined(__APPLE__)
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_statistics64_data_t vmstat;
        mach_port_t host = mach_host_self();

        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vmstat,&count) == KERN_SUCCESS)
        {
            int pageSize;
            size_t size = sizeof(pageSize);
            sysctlbyname("hw.pagesize", &pageSize, &size, nullptr, 0);

            int64_t freeMemBytes = static_cast<int64_t>(vmstat.free_count + vmstat.inactive_count) * pageSize;
            systemMaxMemoryInMB = static_cast<int>(freeMemBytes / (1024 * 1024));
        }
#elif defined(__linux__)
        struct sysinfo info;
        if (sysinfo(&info) == 0)
        {
            systemMaxMemoryInMB = static_cast<int>(info.freeram / (1024 * 1024));
        }
#endif

        return std::min(maxMemoryInMB, systemMaxMemoryInMB);
    }

    // Print the configuration settings
    void printConfig()
    {
        const char *avxLevelStr;
        switch (static_cast<int>(avxLevel))
        {
        case static_cast<int>(AVXLevel::SCALAR):
            avxLevelStr = "SCALAR";
            break;
        case static_cast<int>(AVXLevel::AVX2):
            avxLevelStr = "AVX2";
            break;
        case static_cast<int>(AVXLevel::AVX512):
            avxLevelStr = "AVX512";
            break;
        default:
            avxLevelStr = "UNKNOWN";
            break;
        }

        INFO("OpenMP: %s", openMP ? "enabled" : "disabled");
        INFO("AVX Level: %s", avxLevelStr);
        INFO("Num Threads: %d", numThreads);
        INFO("Max Memory: %d MB", maxMemoryInMB);
    }

    // Member variables to store the configuration
    bool openMP;
    AVXLevel avxLevel;
    int numThreads;
    int maxMemoryInMB;
};

// Function to get the SIMD width based on the AVX level
int getSimdWidth()
{
    ComputeConfig &config = ComputeConfig::getInstance();
    AVXLevel avxLevel = config.getAVXLevel();

    switch (static_cast<int>(avxLevel))
    {
    case static_cast<int>(AVXLevel::AVX512):
#ifndef __APPLE__
        return simd_width_avx512;
#else
        return 0;
#endif
    case static_cast<int>(AVXLevel::AVX2):
        return simd_width_avx2;
    case static_cast<int>(AVXLevel::SCALAR):
    default:
        return 1; // Scalar width
    }
}

int32_t allocateDPTable(int hapLength, int readLength)
{
    ComputeConfig &config = ComputeConfig::getInstance();
    int simdWidth = getSimdWidth();
    int numThreads = config.getNumThreads();

    size_t dp_table_size = (hapLength + 1) * simdWidth * numThreads * sizeof(double);
    size_t transition_size = TRANS_PROB_ARRAY_LENGTH * (readLength + 1) * simdWidth * numThreads * sizeof(double);
    size_t prior_size = (hapLength + 1) * (readLength + 1) * simdWidth * numThreads * sizeof(double);

    DPTable &dpTable = DPTable::getInstance();
    return dpTable.allocate(dp_table_size, transition_size, prior_size);
}

bool initializeNative(OpenMPSetting openMPSetting = OpenMPSetting::FASTEST_AVAILABLE, int numThreads = 1, AVXLevel userAVXLevel = AVXLevel::AVX512, int maxMemoryInMB = 512)
{
    /* Initialize Probability Cache */
    ProbabilityCache &probCache = ProbabilityCache::getInstance();
    int32_t initStatus = probCache.initialize();
    if (initStatus != PDHMM_SUCCESS)
    {
        return false;
    }
    /* Initialize Configuration */
    ComputeConfig &config = ComputeConfig::getInstance();
    config.initialize(openMPSetting, numThreads, userAVXLevel, maxMemoryInMB);

    /* Initialize DP Table based on Configuration */
    int maxHapLength = 500;  // todo: Get maxReadLength from JavaData
    int maxReadLength = 200; // todo: Get maxReadLength from JavaData
    allocateDPTable(maxHapLength, maxReadLength);

    return true;
}

bool doneNative()
{
    return true;
}

int32_t computePDHMM(PDHMMInputData input)
{
    ComputeConfig &config = ComputeConfig::getInstance();
    AVXLevel avxLevel = config.getAVXLevel();
    int numThreads = config.getNumThreads();

    int status = PDHMM_SUCCESS;

    switch (static_cast<int>(avxLevel))
    {
    case static_cast<int>(AVXLevel::AVX512):
        // Call AVX512 implementation
        status = PDHMM_FAILURE;
#ifndef __APPLE__
        status = avx512_impl(input, numThreads);
#endif
        break;

    case static_cast<int>(AVXLevel::AVX2):
        // Call AVX2 implementation
        status = avx2_impl(input, numThreads);
        break;

    case static_cast<int>(AVXLevel::SCALAR):
    default:
        // Call scalar implementation
        status = scalar_impl(input, numThreads);
        break;
    }
    return status;
    // todo: Based on the size of input, call appropriate avx version of implementation
}

int32_t computePDHMM(const int8_t *hap_bases, const int8_t *hap_pdbases, const int8_t *read_bases, const int8_t *read_qual, const int8_t *read_ins_qual, const int8_t *read_del_qual, const int8_t *gcp, double *result, int64_t t, const int64_t *hap_lengths, const int64_t *read_lengths, int32_t maxReadLength, int32_t maxHaplotypeLength)
{
    ComputeConfig &config = ComputeConfig::getInstance();
    AVXLevel avxLevel = config.getAVXLevel();
    int numThreads = config.getNumThreads();

    int status = PDHMM_SUCCESS;

    switch (static_cast<int>(avxLevel))
    {
    case static_cast<int>(AVXLevel::AVX512):
        // Call AVX512 implementation
        status = PDHMM_FAILURE;
#ifndef __APPLE__
        status = computePDHMM_fp_avx512(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
#endif
        break;

    case static_cast<int>(AVXLevel::AVX2):
        // Call AVX2 implementation
        status = computePDHMM_fp_avx2(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
        break;

    case static_cast<int>(AVXLevel::SCALAR):
    default:
        // Call scalar implementation
        status = computePDHMM_serial(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
        break;
    }
    return status;
    // todo: Based on the size of input, call appropriate avx version of implementation
}

#endif // PDHMM_IMPLEMENTATION_H
