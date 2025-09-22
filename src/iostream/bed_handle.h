#include "genomic_interval_handle.h"

#include <functional>
#include <htslib/sam.h>
#include <htslib/tbx.h>
#include <htslib/vcf.h>
#include <memory>
#include <string>

class BedRecord {
public:
  int tid;
  int64_t start;
  int64_t end;
};

class BedHandle : public GenomicIntervalHandle<BedRecord> {
public:
  BedHandle(std::string file_path, ModeType mode, sam_hdr_t *bam_header,
            bcf_hdr_t *vcf_header, int write_compress_level = 0)
      : GenomicIntervalHandle<BedRecord>(file_path, mode),
        bam_header_(bam_header), vcf_header_(vcf_header),
        hts_file_(
            hts_open(file_path.c_str(), mode == ModeType::READ ? "r" : "w"),
            [](htsFile *file) {
              if (file)
                hts_close(file);
            }),
        tbx_(tbx_index_load(file_path.c_str()), [](tbx_t *tbx) {
          if (tbx)
            tbx_destroy(tbx);
        }) {
    if (!hts_file_) {
      throw std::runtime_error("Failed to open BED file: " + file_path);
    }
    if (mode == ModeType::WRITE) {
      hts_set_opt(hts_file_.get(), HTS_OPT_COMPRESSION_LEVEL,
                  write_compress_level);
    }
    if (!bam_header_ && !vcf_header_) {
      throw std::runtime_error("BAM or VCF header must be provided");
    }
  }

  htsFile *get_htslib_handle() override { return hts_file_.get(); }

  void write_record(BedRecord *record) override;

  bool has_next() override;

  bool next(BedRecord *record) override;

  void close() override;

  void query_from_region(const char *region) override;

  void query_from_region(int tid, int64_t start, int64_t end) override;

  void query_from_multi_region(char **regions, int num_regions) override;

  ~BedHandle() {
    hts_file_.reset();
    tbx_.reset();
    hts_itr_.reset();
  }

private:
  std::unique_ptr<htsFile, std::function<void(htsFile *)>> hts_file_;
  std::unique_ptr<tbx_t, std::function<void(tbx_t *)>> tbx_;
  std::unique_ptr<hts_itr_t, std::function<void(hts_itr_t *)>> hts_itr_;
  sam_hdr_t *bam_header_;
  bcf_hdr_t *vcf_header_;
  int ret_code_;
};