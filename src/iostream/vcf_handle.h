#include "genomic_interval_handle.h"

#include <functional>
#include <htslib/hts.h>
#include <htslib/tbx.h>
#include <htslib/vcf.h>
#include <string>

class VcfHandle : public GenomicIntervalHandle<bcf1_t> {
public:
  VcfHandle(std::string file_path, ModeType mode, int write_compress_level = 0)
      : GenomicIntervalHandle<bcf1_t>(file_path, mode),
        hts_file_(
            vcf_open(file_path.c_str(), mode == ModeType::READ ? "r" : "w"),
            [](htsFile *file) {
              if (file)
                vcf_close(file);
            }),
        header_(nullptr), hts_itr_(nullptr, [](hts_itr_t *itr) {
          if (itr)
            hts_itr_destroy(itr);
        }) {
    if (!hts_file_) {
      throw std::runtime_error("Failed to open VCF file: " + file_path);
    }

    if (mode == ModeType::WRITE) {
      hts_set_opt(hts_file_.get(), HTS_OPT_COMPRESSION_LEVEL,
                  write_compress_level);
    } else {
      header_ = bcf_hdr_read(hts_file_.get());
      // 尝试自动加载索引
      hts_idx_ = std::unique_ptr<tbx_t, std::function<void(tbx_t *)>>(
          tbx_index_load(file_path.c_str()), [](tbx_t *idx) {
            if (idx)
              tbx_destroy(idx);
          });
    }
  }
  htsFile *get_htslib_handle() override { return hts_file_.get(); }

  void write_record(bcf1_t *record) override;
  bool has_next() override;
  bool next(bcf1_t *record) override;
  void close() override {
    hts_idx_.reset();
    hts_itr_.reset();
    hts_file_.reset();
    if (header_) {
      bcf_hdr_destroy(header_);
    }
  }
  void query_from_region(const char *region) override;
  void query_from_region(int tid, int64_t start, int64_t end) override;
  void query_from_multi_region(char **regions, int num_regions) override {
    throw std::runtime_error(
        "query_from_multi_region is not supported for VCF");
  }
  ~VcfHandle() {
    hts_idx_.reset();
    hts_itr_.reset();
    hts_file_.reset();
    if (header_) {
      bcf_hdr_destroy(header_);
    }
  }

private:
  std::unique_ptr<htsFile, std::function<void(htsFile *)>> hts_file_;
  bcf_hdr_t *header_;
  std::unique_ptr<hts_itr_t, std::function<void(hts_itr_t *)>> hts_itr_;
  std::unique_ptr<tbx_t, std::function<void(tbx_t *)>> hts_idx_;

  int ret_code_;
};