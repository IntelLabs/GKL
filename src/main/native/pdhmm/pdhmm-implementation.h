#ifndef PDHMM_IMPLEMENTATION_H
#define PDHMM_IMPLEMENTATION_H

#include "pdhmm-common.h"
#include "pdhmm-serial.h"
#include "avx2_impl.h"
#ifndef __APPLE__
#include "avx512_impl.h"
#endif

#ifdef linux
#include <omp.h>
#endif
#include <sys/sysinfo.h>

enum class AVXLevel
{
    SCALAR,
    AVX2,
    AVX512
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
    void initialize(int numThreads = 1, AVXLevel userAVXLevel = AVXLevel::AVX512, int maxMemoryInMB = 512)
    {
        this->openMP = (numThreads > 1) && isOpenMPAvailable();
        this->numThreads = this->openMP ? getBestAvailableNumThreads(numThreads) : 1;
        this->avxLevel = getBestAvailableAVXLevel(userAVXLevel);
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
    ComputeConfig() : openMP(false), avxLevel(AVXLevel::SCALAR), numThreads(1) {}

    // Private destructor
    ~ComputeConfig() {}

    // Delete copy constructor and assignment operator to prevent copying
    ComputeConfig(const ComputeConfig &) = delete;
    ComputeConfig &operator=(const ComputeConfig &) = delete;

    // Detect the system's best available AVX capabilities
    AVXLevel detectBestAVXLevel()
    {
        if (is_avx512_supported())
        {
            return AVXLevel::AVX512;
        }
        else if (is_avx_supported() && is_avx2_supported() && is_sse_supported())
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
        if (userAVXLevel == AVXLevel::SCALAR)
        {
            return AVXLevel::SCALAR;
        }
        else if (userAVXLevel == AVXLevel::AVX512 && is_avx512_supported())
        {
            return AVXLevel::AVX512;
        }
        else if (userAVXLevel == AVXLevel::AVX2 && is_avx_supported() && is_avx2_supported() && is_sse_supported())
        {
            return AVXLevel::AVX2;
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

    int getMaxMemoryAvailable(int maxMemoryInMB)
    {
        struct sysinfo info;
        if (sysinfo(&info) != 0)
        {
            // If sysinfo fails, return the provided maxMemoryInMB
            return maxMemoryInMB;
        }
        // Convert available memory from bytes to megabytes
        int systemMaxMemoryInMB = static_cast<int>(info.freeram / (1024 * 1024));

        // Return the minimum of the provided maxMemoryInMB and the system's available memory
        return std::min(maxMemoryInMB, systemMaxMemoryInMB);
    }

    void printConfig()
    {
        const char *avxLevelStr;
        switch (avxLevel)
        {
        case AVXLevel::SCALAR:
            avxLevelStr = "SCALAR";
            break;
        case AVXLevel::AVX2:
            avxLevelStr = "AVX2";
            break;
        case AVXLevel::AVX512:
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

    switch (avxLevel)
    {
    case AVXLevel::AVX512:
        return simd_width_avx512;
    case AVXLevel::AVX2:
        return simd_width_avx2;
    case AVXLevel::SCALAR:
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

bool initializeNative(int numThreads = 1, AVXLevel userAVXLevel = AVXLevel::AVX512, int maxMemoryInMB = 512)
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
    config.initialize(numThreads, userAVXLevel, maxMemoryInMB);

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

    switch (avxLevel)
    {
    case AVXLevel::AVX512:
        // Call AVX512 implementation
        status = avx512_impl(input, numThreads);
        break;

    case AVXLevel::AVX2:
        // Call AVX2 implementation
        status = avx2_impl(input, numThreads);
        break;

    case AVXLevel::SCALAR:
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

    switch (avxLevel)
    {
    case AVXLevel::AVX512:
        // Call AVX512 implementation
        status = computePDHMM_fp_avx512(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
        break;

    case AVXLevel::AVX2:
        // Call AVX2 implementation
        status = computePDHMM_fp_avx2(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
        break;

    case AVXLevel::SCALAR:
    default:
        // Call scalar implementation
        status = computePDHMM_serial(hap_bases, hap_pdbases, read_bases, read_qual, read_ins_qual, read_del_qual, gcp, result, t, hap_lengths, read_lengths, maxReadLength, maxHaplotypeLength, numThreads);
        break;
    }
    return status;
    // todo: Based on the size of input, call appropriate avx version of implementation
}

#endif // PDHMM_IMPLEMENTATION_H
