#include "genomic_interval_handle.h"
#include <functional>
#include <htslib/sam.h>
#include <memory>

class BamHandle : public GenomicIntervalHandle<bam1_t> {
public:
  BamHandle(std::string file_path, ModeType mode, int write_compress_level = 0)
      : GenomicIntervalHandle<bam1_t>(file_path, mode),
        hts_file_(
            sam_open(file_path.c_str(), mode == ModeType::READ ? "r" : "w"),
            [](htsFile *file) {
              if (file)
                sam_close(file);
            }),
        header_(nullptr), write_compress_level_(write_compress_level),
        hts_idx_(nullptr,
                 [](hts_idx_t *idx) {
                   if (idx)
                     hts_idx_destroy(idx);
                 }),
        hts_itr_(nullptr, [](hts_itr_t *itr) {
          if (itr)
            hts_itr_destroy(itr);
        }) {
    if (!hts_file_) {
      throw std::runtime_error("Failed to open BAM file: " + file_path);
    }

    if (mode == ModeType::WRITE) {
      hts_set_opt(hts_file_.get(), HTS_OPT_COMPRESSION_LEVEL,
                  write_compress_level_);
    } else {
      header_ = sam_hdr_read(hts_file_.get());
      // 尝试自动加载索引
      load_index();
    }
  }

  htsFile *get_htslib_handle() override { return hts_file_.get(); }

  void write_record(bam1_t *record) override;

  bool has_next() override;

  bool next(bam1_t *record) override;

  void close() override {
    hts_idx_.reset();
    hts_itr_.reset();
    hts_file_.reset();
  }

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

  // 区间查询相关方法
  void load_index(const std::string &index_path = "");
  void query_from_region(const char *region) override;
  void query_from_region(int tid, int64_t start, int64_t end) override;
  void query_from_multi_region( char **regions , int num_regions ) override;

private:
  std::unique_ptr<htsFile, std::function<void(htsFile *)>> hts_file_;
  int write_compress_level_;
  sam_hdr_t *header_;
  int ret_code_;
  std::unique_ptr<hts_idx_t, std::function<void(hts_idx_t *)>> hts_idx_;
  std::unique_ptr<hts_itr_t, std::function<void(hts_itr_t *)>> hts_itr_;
};