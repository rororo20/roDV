#include <htslib/sam.h>
#include <iomanip>

#include "genomic_interval_handle.h"

class BamHandle : public GenomicIntervalHandle<bam1_t> {
public:
  BamHandle(std::string file_path, ModeType mode, int write_compress_level = 0)
      : GenomicIntervalHandle<bam1_t>(file_path, mode), hts_file_(nullptr),
        header_(nullptr), write_compress_level_(write_compress_level) {
    hts_file_ = sam_open(file_path.c_str(), mode == ModeType::READ ? "r" : "w");

    if (mode == ModeType::WRITE) {
      hts_set_opt(hts_file_, HTS_OPT_COMPRESSION_LEVEL, write_compress_level_);
    } else {
      header_ = sam_hdr_read(hts_file_);
    }
  }

  htsFile *get_htslib_handle() override { return hts_file_; }

  void write_record(bam1_t *record) override;

  bool has_next() override;

  bool next(bam1_t *record) override;

  void close() override { sam_close(hts_file_); }

  void set_header(sam_hdr_t *header) {
    if (get_mode() == ModeType::READ) {
      throw std::runtime_error("mode is not WRITE");
    }
    if (header_ == nullptr) {
      header_ = header;

    } else {
      throw std::runtime_error("header is already set");
    }
  }

private:
  htsFile *hts_file_;
  int write_compress_level_;
  sam_hdr_t *header_;
  int ret_code_;
};