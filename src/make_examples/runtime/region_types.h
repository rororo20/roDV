#ifndef RODV_MAKE_EXAMPLES_REGION_TYPES_H_
#define RODV_MAKE_EXAMPLES_REGION_TYPES_H_

#include "snp_counter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rodv {
namespace make_examples {

// One worker owns one RegionResource for the whole region. Reads are bam_dup
// copies; the BAM file handle never lives here.
struct RegionResource {
  int id = 0;
  std::vector<bam1_t *> reads;
  // DeepTrio parent reads for this same interval. Empty unless a parent
  // ReadSource was supplied.
  std::vector<bam1_t *> parent_reads;

  void reset() {
    for (bam1_t *read : reads) {
      bam_destroy1(read);
    }
    for (bam1_t *read : parent_reads) {
      bam_destroy1(read);
    }
    reads.clear();
    parent_reads.clear();
  }

  ~RegionResource() { reset(); }

  RegionResource() = default;
  RegionResource(const RegionResource &) = delete;
  RegionResource &operator=(const RegionResource &) = delete;
};

// One interval handed from the dispatch thread to a worker.
struct RegionSource {
  int64_t source_id = 0;
  int tid = -1;
  int64_t start = 0;
  int64_t end = 0;
  std::string ref;
  std::shared_ptr<RegionResource> resource;
  bool finish = false;
};

struct TruthVariant {
  int tid = 0;
  int64_t pos = 0;
  std::string ref;
  std::string alt;
  int genotype = -1; // 0 hom-ref, 1 het, 2 hom-alt
};

struct RefBlock {
  int64_t start = 0;
  int64_t end = 0;
  int depth = 0;
};

// Ordered writer input. finish=true carries the number of real regions in
// source_id and is not written.
struct ExampleRecord {
  int64_t source_id = 0;
  int64_t start = 0;
  std::vector<Candidate> candidates;
  std::vector<int> labels;
  std::vector<RefBlock> ref_blocks;
  std::vector<uint8_t> tensors;
  bool finish = false;
};

} // namespace make_examples
} // namespace rodv

#endif
