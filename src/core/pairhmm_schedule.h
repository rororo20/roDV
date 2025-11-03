

#ifndef PAIRHMM_SCHEDULE_H_
#define PAIRHMM_SCHEDULE_H_
#include "pairhmm/common/common.h"
#include <cstdint>
#include <vector>

using namespace pairhmm::common;

namespace pairhmm {
namespace schedule {

bool schedule_pairhmm(
    const std::vector<std::vector<uint8_t>> &haplotypes,
    const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result,
    const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities,
    const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities,
    bool use_double = false,
    double max_idle_ratio_float = 0.1,
    double max_idle_ratio_double = 0.1,
    bool verbose = false);
}
} // namespace pairhmm

#endif