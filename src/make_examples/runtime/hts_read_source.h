#ifndef RODV_MAKE_EXAMPLES_HTS_READ_SOURCE_H_
#define RODV_MAKE_EXAMPLES_HTS_READ_SOURCE_H_

#include "runtime/read_dispatch.h"

#include <htslib/sam.h>
#include <htslib/thread_pool.h>

#include <string>

namespace rodv {
namespace make_examples {

// Coordinate-sorted BAM or CRAM. decode_threads is the htslib decompression
// pool and is not one of the compute workers. The bam1_t returned by next()
// is owned here and is invalid after the following next() or destruction.
// Only the dispatch thread may call next().
class HtsReadSource : public ReadSource {
public:
  HtsReadSource(const std::string &path, int decode_threads,
                const std::string &fasta_for_cram = {});
  ~HtsReadSource() override;

  bool next(int *tid, bam1_t **read) override;

private:
  htsFile *file_ = nullptr;
  sam_hdr_t *header_ = nullptr;
  hts_tpool *pool_ = nullptr;
  htsThreadPool thread_pool_{};
  bam1_t *buffer_ = nullptr;
};

} // namespace make_examples
} // namespace rodv

#endif
