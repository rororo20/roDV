#ifndef RODV_MAKE_EXAMPLES_PILEUP_STAGE_H_
#define RODV_MAKE_EXAMPLES_PILEUP_STAGE_H_

#include "runtime/region_types.h"

#include <cstdint>
#include <vector>

namespace rodv {
namespace make_examples {

// Pixel encoding is a later module. The default stage returns no bytes so the
// pipeline can be tested without a pileup image.
class PileupStage {
public:
  virtual ~PileupStage() = default;
  virtual std::vector<uint8_t>
  encode(int64_t source_id, int64_t ref_start,
         const std::vector<Candidate> &candidates, const std::string &ref,
         const std::vector<bam1_t *> &reads, const std::vector<bam1_t *> &parents,
         bool phase) {
    (void)source_id;
    (void)ref_start;
    (void)candidates;
    (void)ref;
    (void)reads;
    (void)parents;
    (void)phase;
    return {};
  }
};

} // namespace make_examples
} // namespace rodv

#endif
