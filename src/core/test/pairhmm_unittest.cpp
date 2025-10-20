#include "../pairhmm/intra/pairhmm_api.h"
#include <cmath>
#include <cpuid.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>
#include <x86intrin.h>

using namespace pairhmm::intra;

inline int check_xcr0_ymm()
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & 6) == 6);
}

// helper function
inline int check_xcr0_zmm()
{
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
#if defined(_MSC_VER)
    /* min VS2010 SP1 compiler is required */
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    /* check if xmm, zmm and zmm state are enabled in XCR0 */
    return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm);
}


/**
 * @brief PairHMM Unit Tests using GoogleTest
 * 参考 Intel GKL PairHMM 测试用例
 *
 * 测试数据格式:
 * hap-bases read-bases read-qual read-ins-qual read-del-qual gcp
 * expected-result
 */

/**
 * @brief CPU指令集检测工具
 */
class CpuFeatureDetector {
public:
  /**
   * @brief 检测当前CPU是否支持AVX512指令集
   * @return true 如果支持AVX512，false 否则
   */
  static bool hasAVX512Support() {
    static bool checked = false;
    static bool result = false;

    if (!checked) {
      result = checkAVX512Support();
      checked = true;
    }

    return result;
  }

private:
  /**
   * @brief 实际检测AVX512支持的实现
   */
  static bool checkAVX512Support() {
#ifndef __APPLE__
    uint32_t a, b, c, d;
    uint32_t osxsave_mask = (1 << 27);     // OSX.
    uint32_t avx512_skx_mask = (1 << 16) | // AVX-512F
                               (1 << 17) | // AVX-512DQ
                               (1 << 30) | // AVX-512BW
                               (1 << 31);  // AVX-512VL

    // step 1 - must ensure OS supports extended processor state management
    __cpuid_count(1, 0, a, b, c, d);
    if ((c & osxsave_mask) != osxsave_mask) {
      return true;
    }

    // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
    if (!check_xcr0_zmm()) {
      return false;
    }

    // step 3 - must ensure AVX512 is supported
    __cpuid_count(7, 0, a, b, c, d);
    if ((b & avx512_skx_mask) != avx512_skx_mask) {
      return false;
    }

    return true;
#else
    return false;
#endif
  }
};

/**
 * @brief 测试用例数据结构
 */
struct TestCaseData {
  std::string hap_bases;
  std::string read_bases;
  std::vector<uint8_t> read_qual;
  std::vector<uint8_t> read_ins_qual;
  std::vector<uint8_t> read_del_qual;
  std::vector<uint8_t> gcp;
  double expected_result;
  uint32_t line_number;
};

/**
 * @brief 测试数据解析器
 */
class TestDataParser {
public:
  /**
   * @brief 从文件加载测试数据
   */
  static std::vector<TestCaseData> loadTestData(const std::string &filename) {
    std::vector<TestCaseData> test_cases;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test data file: " + filename);
    }

    std::string line;
    uint32_t line_num = 0;

    // 跳过第一行（标题行）
    if (std::getline(file, line)) {
      line_num++;
    }

    while (std::getline(file, line)) {
      line_num++;

      std::istringstream iss(line);
      TestCaseData data;
      std::string hap_bases, read_bases, read_qual_str, read_ins_qual_str;
      std::string read_del_qual_str, gcp_str, expected_str;

      if (iss >> hap_bases >> read_bases >> read_qual_str >>
          read_ins_qual_str >> read_del_qual_str >> gcp_str >> expected_str) {

        data.hap_bases = hap_bases;
        data.read_bases = read_bases;
        data.read_qual = parseQualityString(read_qual_str);
        data.read_ins_qual = parseQualityString(read_ins_qual_str);
        data.read_del_qual = parseQualityString(read_del_qual_str);
        data.gcp = parseQualityString(gcp_str);
        data.expected_result = std::stod(expected_str);
        data.line_number = line_num;

        test_cases.push_back(data);
      }
    }

    return test_cases;
  }

private:
  /**
   * @brief 解析质量字符串为uint8_t数组
   */
  static std::vector<uint8_t> parseQualityString(const std::string &qual_str) {
    std::vector<uint8_t> qual;
    for (char c : qual_str) {
      qual.push_back(static_cast<uint8_t>(c));
    }
    return qual;
  }
};

/**
 * @brief 对齐内存分配器
 */
template <size_t Alignment> class AlignedAllocator {
public:
  using value_type = uint8_t;
  static constexpr size_t align_value = Alignment;

  uint8_t *allocate(size_t n) {
    size_t size = n + Alignment - 1;
    void *ptr = nullptr;

// 使用 posix_memalign 进行对齐分配
#ifdef _GNU_SOURCE
    if (posix_memalign(&ptr, Alignment, size) != 0) {
      ptr = nullptr;
    }
#else
    // 回退到普通的 malloc
    ptr = std::malloc(size);
#endif

    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<uint8_t *>(ptr);
  }

  void deallocate(uint8_t *ptr, size_t) {
    if (ptr) {
      std::free(ptr);
    }
  }
};

/**
 * @brief 测试用例包装器，确保内存安全
 */
class TestCaseWrapper {
public:
  TestCaseWrapper(const TestCaseData &data)
      : hap_data_(nullptr), rs_data_(nullptr), q_data_(nullptr),
        i_data_(nullptr), d_data_(nullptr), c_data_(nullptr) {
    // 使用对齐分配器分配内存
    size_t hap_size = data.hap_bases.size() + 32;
    size_t rs_size = data.read_bases.size() + 32;
    size_t q_size = data.read_qual.size() + 32;
    size_t i_size = data.read_ins_qual.size() + 32;
    size_t d_size = data.read_del_qual.size() + 32;
    size_t c_size = data.gcp.size() + 32;

    hap_data_ = allocator_.allocate(hap_size);
    rs_data_ = allocator_.allocate(rs_size);
    q_data_ = allocator_.allocate(q_size);
    i_data_ = allocator_.allocate(i_size);
    d_data_ = allocator_.allocate(d_size);
    c_data_ = allocator_.allocate(c_size);

    // 填充数据
    for (size_t i = 0; i < data.hap_bases.size(); i++) {
      hap_data_[i] = data.hap_bases[i];
    }

    for (size_t i = 0; i < data.read_bases.size(); i++) {
      rs_data_[i] = data.read_bases[i];
    }

    std::copy(data.read_qual.begin(), data.read_qual.end(), q_data_);
    std::copy(data.read_ins_qual.begin(), data.read_ins_qual.end(), i_data_);
    std::copy(data.read_del_qual.begin(), data.read_del_qual.end(), d_data_);
    std::copy(data.gcp.begin(), data.gcp.end(), c_data_);

    // 设置TestCase结构
    tc_.haplen = static_cast<uint32_t>(data.hap_bases.size());
    tc_.rslen = static_cast<uint32_t>(data.read_bases.size());
    tc_.hap = hap_data_;
    tc_.rs = rs_data_;
    tc_.q = q_data_;
    tc_.i = i_data_;
    tc_.d = d_data_;
    tc_.c = c_data_;
  }

  ~TestCaseWrapper() {
    allocator_.deallocate(hap_data_, 0);
    allocator_.deallocate(rs_data_, 0);
    allocator_.deallocate(q_data_, 0);
    allocator_.deallocate(i_data_, 0);
    allocator_.deallocate(d_data_, 0);
    allocator_.deallocate(c_data_, 0);
  }

  // 禁用拷贝构造和赋值操作
  TestCaseWrapper(const TestCaseWrapper &) = delete;
  TestCaseWrapper &operator=(const TestCaseWrapper &) = delete;

  const TestCase &getTestCase() const { return tc_; }

private:
  AlignedAllocator<32> allocator_;
  uint8_t *hap_data_;
  uint8_t *rs_data_;
  uint8_t *q_data_;
  uint8_t *i_data_;
  uint8_t *d_data_;
  uint8_t *c_data_;
  TestCase tc_;
};

/**
 * @brief 测试基类，包含通用测试逻辑
 */
class PairHMMTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    // 尝试多个可能的路径
    std::vector<std::string> possible_paths = {
        "../../../../resouces/pairhmm-testdata.txt",
        "../../../resouces/pairhmm-testdata.txt",
        "../../resouces/pairhmm-testdata.txt", "resouces/pairhmm-testdata.txt",
        "../resouces/pairhmm-testdata.txt"};

    bool data_loaded = false;
    for (const auto &path : possible_paths) {
      try {
        test_data_ = TestDataParser::loadTestData(path);
        if (!test_data_.empty()) {
          data_loaded = true;
          break;
        }
      } catch (const std::exception &) {
        continue;
      }
    }

    ASSERT_TRUE(data_loaded) << "Cannot load test data from any expected path";
  }

  /**
   * @brief 检查结果精度是否在合理范围内
   */
  template <typename T>
  void checkResultAccuracy(T actual, double expected,
                           const std::string &test_name) {
    double actual_d = static_cast<double>(actual);
    double rel_error = std::abs(actual_d - expected) / std::abs(expected);

    // 对于float，相对误差应该小于1e-5
    // 对于double，相对误差应该小于1e-10
    double tolerance = std::is_same<T, float>::value ? 1e-5 : 1e-10;

    EXPECT_LT(rel_error, tolerance)
        << "Test: " << test_name << ", Expected: " << expected
        << ", Actual: " << actual_d << ", Relative error: " << rel_error;
  }

  std::vector<TestCaseData> test_data_;
};

/**
 * @brief AVX2 Float 测试
 */
class PairHMMAVX2FloatTest : public PairHMMTestBase {};

TEST_F(PairHMMAVX2FloatTest, AllTestCases) {
  for (const auto &data : test_data_) {
    TestCaseWrapper wrapper(data);
    float result = compute_pairhmm_avx2_float(wrapper.getTestCase());

    SCOPED_TRACE("Line " + std::to_string(data.line_number));
    checkResultAccuracy(result, data.expected_result, "AVX2 Float");
  }
}

/**
 * @brief AVX2 Double 测试
 */
class PairHMMAVX2DoubleTest : public PairHMMTestBase {};

TEST_F(PairHMMAVX2DoubleTest, AllTestCases) {
  for (const auto &data : test_data_) {
    TestCaseWrapper wrapper(data);
    double result = compute_pairhmm_avx2_double(wrapper.getTestCase());

    SCOPED_TRACE("Line " + std::to_string(data.line_number));
    checkResultAccuracy(result, data.expected_result, "AVX2 Double");
  }
}

/**
 * @brief AVX512 Float 测试
 */
class PairHMMAVX512FloatTest : public PairHMMTestBase {
protected:
  void SetUp() override {
    PairHMMTestBase::SetUp();
    if (!CpuFeatureDetector::hasAVX512Support()) {
      GTEST_SKIP() << "AVX512 not supported on this system";
    }
  }
};

TEST_F(PairHMMAVX512FloatTest, AllTestCases) {
  if (!CpuFeatureDetector::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (const auto &data : test_data_) {
    TestCaseWrapper wrapper(data);
    float result = compute_pairhmm_avx512_float(wrapper.getTestCase());

    SCOPED_TRACE("Line " + std::to_string(data.line_number));
    checkResultAccuracy(result, data.expected_result, "AVX512 Float");
  }
}

/**
 * @brief AVX512 Double 测试
 */
class PairHMMAVX512DoubleTest : public PairHMMTestBase {
protected:
  void SetUp() override {
    PairHMMTestBase::SetUp();
    if (!CpuFeatureDetector::hasAVX512Support()) {
      GTEST_SKIP() << "AVX512 not supported on this system";
    }
  }
};

TEST_F(PairHMMAVX512DoubleTest, AllTestCases) {
  if (!CpuFeatureDetector::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (const auto &data : test_data_) {
    TestCaseWrapper wrapper(data);
    double result = compute_pairhmm_avx512_double(wrapper.getTestCase());

    SCOPED_TRACE("Line " + std::to_string(data.line_number));
    checkResultAccuracy(result, data.expected_result, "AVX512 Double");
  }
}

/**
 * @brief 精度一致性测试
 */
class PairHMMPrecisionConsistencyTest : public PairHMMTestBase {};

TEST_F(PairHMMPrecisionConsistencyTest, AVX2FloatVsDouble) {
  for (size_t i = 0; i < std::min(test_data_.size(), size_t(10)); ++i) {
    const auto &data = test_data_[i];
    TestCaseWrapper wrapper(data);

    float result_float = compute_pairhmm_avx2_float(wrapper.getTestCase());
    double result_double = compute_pairhmm_avx2_double(wrapper.getTestCase());

    double rel_error =
        std::abs(result_float - result_double) / std::abs(result_double);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " - AVX2 Float vs Double");
    EXPECT_LT(rel_error, 1e-5)
        << "Float: " << result_float << ", Double: " << result_double
        << ", Relative error: " << rel_error;
  }
}

TEST_F(PairHMMPrecisionConsistencyTest, AVX512FloatVsDouble) {
  if (!CpuFeatureDetector::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (size_t i = 0; i < std::min(test_data_.size(), size_t(10)); ++i) {
    const auto &data = test_data_[i];
    TestCaseWrapper wrapper(data);

    float result_float = compute_pairhmm_avx512_float(wrapper.getTestCase());
    double result_double = compute_pairhmm_avx512_double(wrapper.getTestCase());

    double rel_error =
        std::abs(result_float - result_double) / std::abs(result_double);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " - AVX512 Float vs Double");
    EXPECT_LT(rel_error, 1e-5)
        << "Float: " << result_float << ", Double: " << result_double
        << ", Relative error: " << rel_error;
  }
}

/**
 * @brief 指令集一致性测试
 */
class PairHMMInstructionSetConsistencyTest : public PairHMMTestBase {};

TEST_F(PairHMMInstructionSetConsistencyTest, AVX2VsAVX512Float) {
  if (!CpuFeatureDetector::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (size_t i = 0; i < std::min(test_data_.size(), size_t(10)); ++i) {
    const auto &data = test_data_[i];
    TestCaseWrapper wrapper(data);

    float result_avx2 = compute_pairhmm_avx2_float(wrapper.getTestCase());
    float result_avx512 = compute_pairhmm_avx512_float(wrapper.getTestCase());

    double rel_error =
        std::abs(result_avx2 - result_avx512) / std::abs(result_avx512);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " - AVX2 vs AVX512 Float");
    EXPECT_LT(rel_error, 1e-5)
        << "AVX2: " << result_avx2 << ", AVX512: " << result_avx512
        << ", Relative error: " << rel_error;
  }
}

TEST_F(PairHMMInstructionSetConsistencyTest, AVX2VsAVX512Double) {
  if (!CpuFeatureDetector::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (size_t i = 0; i < std::min(test_data_.size(), size_t(10)); ++i) {
    const auto &data = test_data_[i];
    TestCaseWrapper wrapper(data);

    double result_avx2 = compute_pairhmm_avx2_double(wrapper.getTestCase());
    double result_avx512 = compute_pairhmm_avx512_double(wrapper.getTestCase());

    double rel_error =
        std::abs(result_avx2 - result_avx512) / std::abs(result_avx512);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " - AVX2 vs AVX512 Double");
    EXPECT_LT(rel_error, 1e-10)
        << "AVX2: " << result_avx2 << ", AVX512: " << result_avx512
        << ", Relative error: " << rel_error;
  }
}

/**
 * @brief 测试程序入口点
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // 打印系统信息
  if (CpuFeatureDetector::hasAVX512Support()) {
    std::cout << "AVX512 support detected - running all tests\n";
  } else {
    std::cout << "AVX512 not supported - skipping AVX512 tests\n";
  }

  return RUN_ALL_TESTS();
}
