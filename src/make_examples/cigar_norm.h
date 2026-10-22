#ifndef RODV_MAKE_EXAMPLES_CIGAR_NORM_H_
#define RODV_MAKE_EXAMPLES_CIGAR_NORM_H_

#include <htslib/sam.h>

#include <cstdint>
#include <string>

namespace rodv {
namespace make_examples {

// Shift insertions and deletions left through a repeat, the same normalization
// DeepVariant applies before counting. Rewrites the cigar on read. Returns
// true when the cigar changed. ref[0] is the base at ref_start.
bool left_align_cigar(bam1_t *read, int64_t ref_start, const std::string &ref);

} // namespace make_examples
} // namespace rodv

#endif
