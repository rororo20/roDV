#ifndef RODV_MAKE_EXAMPLES_SNP_REGION_H_
#define RODV_MAKE_EXAMPLES_SNP_REGION_H_

#include "read_filter.h"
#include "realigner.h"
#include "snp_counter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rodv {
namespace make_examples {

// One interval handed to a worker. Reads are already filtered copies; the
// worker does not touch the BAM. ref[0] is the base at start.
struct SnpRegionInput {
  int64_t start = 0;
  std::string ref;
  std::vector<const bam1_t *> reads;
};

struct RegionSnpResult {
  int64_t start = 0;
  std::vector<Candidate> candidates;
};

// Realigns each region, then counts alleles on the new cigars. Results follow
// the input order, not completion order. nthreads uses hc's [1, 128] clamp.
// SNP, indel, and realigner thresholds are the r1.6.1 defaults.
std::vector<RegionSnpResult>
count_snp_regions(const std::vector<SnpRegionInput> &regions, int nthreads,
                  const ReadFilter &filter = {}, int min_base_quality = 10,
                  const CandidateThresholds &thresholds = {},
                  const RealignerOptions &realigner = {});

} // namespace make_examples
} // namespace rodv

#endif
