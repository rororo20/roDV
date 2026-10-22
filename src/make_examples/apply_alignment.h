#ifndef RODV_MAKE_EXAMPLES_APPLY_ALIGNMENT_H_
#define RODV_MAKE_EXAMPLES_APPLY_ALIGNMENT_H_

#include "realigner.h"

#include <memory>

namespace rodv {
namespace make_examples {

struct BamDeleter {
  void operator()(bam1_t *read) const { bam_destroy1(read); }
};

using BamPtr = std::unique_ptr<bam1_t, BamDeleter>;

// Copies src onto the realigned cigar when that cigar consumes the same
// number of query bases. Returns null when the read should be counted as-is.
BamPtr with_alignment(const bam1_t *src, const AlignedRead &aligned);

} // namespace make_examples
} // namespace rodv

#endif
