#include "runtime/hts_read_source.h"

#include <stdexcept>

namespace rodv {
namespace make_examples {

HtsReadSource::HtsReadSource(const std::string &path, int decode_threads,
                             const std::string &fasta_for_cram)
    : file_(hts_open(path.c_str(), "r")), buffer_(bam_init1()) {
  if (file_ == nullptr || buffer_ == nullptr) {
    throw std::runtime_error("failed to open reads: " + path);
  }
  if (decode_threads > 0) {
    pool_ = hts_tpool_init(decode_threads);
    if (pool_ == nullptr) {
      throw std::runtime_error("failed to create hts decompression pool");
    }
    thread_pool_.pool = pool_;
    thread_pool_.qsize = 0;
    if (hts_set_thread_pool(file_, &thread_pool_) < 0) {
      throw std::runtime_error("failed to attach hts decompression pool");
    }
  }
  if (!fasta_for_cram.empty()) {
    hts_set_opt(file_, CRAM_OPT_REFERENCE, fasta_for_cram.c_str());
  }
  header_ = sam_hdr_read(file_);
  if (header_ == nullptr) {
    throw std::runtime_error("failed to read BAM/CRAM header: " + path);
  }
}

HtsReadSource::~HtsReadSource() {
  bam_destroy1(buffer_);
  if (header_ != nullptr) {
    sam_hdr_destroy(header_);
  }
  if (file_ != nullptr) {
    hts_close(file_);
  }
  if (pool_ != nullptr) {
    hts_tpool_destroy(pool_);
  }
}

bool HtsReadSource::next(int *tid, bam1_t **read) {
  const int rc = sam_read1(file_, header_, buffer_);
  if (rc < 0) {
    return false;
  }
  *tid = buffer_->core.tid;
  *read = buffer_;
  return true;
}

} // namespace make_examples
} // namespace rodv
