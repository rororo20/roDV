#include "bed_handle.h"
#include "string.h"
#include <htslib/bgzf.h>
#include <htslib/hfile.h>
#include <htslib/hts.h>
#include <htslib/tbx.h>
#include <sstream>
#include <stdexcept>
#include <htslib/kseq.h>

void BedHandle::init_chrom_info() {
  if (bam_header_) {
    // 从BAM header获取染色体信息
    chrom_info_.reserve(bam_header_->n_targets);

    for (int i = 0; i < bam_header_->n_targets; ++i) {
      chrom_info_.emplace_back(bam_header_->target_name[i], strlen(bam_header_->target_name[i]),bam_header_->target_len[i]);
    }
  } else if (vcf_header_) {
    // 从VCF header获取染色体信息
    int n_contigs = vcf_header_->n[BCF_DT_CTG];
    chrom_info_.reserve(n_contigs);

    for (int i = 0; i < n_contigs; ++i) {
      const char *name = bcf_hdr_id2name(vcf_header_, i);
      int length = bcf_hdr_id2length(vcf_header_, BCF_HT_INT, i);
      if (name) {
        chrom_info_.emplace_back(name, strlen(name), length);
      }
    }
  }
}

void BedHandle::write_record(BedRecord *record) {
  // 获取染色体名称
  std::string chrom_name;
  if (record->tid >= 0 && record->tid < static_cast<int>(chrom_info_.size())) {
    chrom_name = chrom_info_[record->tid].name;
  } else {
    throw std::runtime_error("Invalid chromosome ID: " +
                             std::to_string(record->tid));
  }

  // 构建输出字符串
  std::ostringstream oss;
  oss << chrom_name << "\t" << record->start << "\t" << record->end << "\n";
  std::string record_str = oss.str();

  // 检查文件格式
  const htsFormat *format = hts_get_format(hts_file_.get());
  if (format && format->compression == bgzf) {
    // BGZF格式，使用bgzf_write
    BGZF *bgzf_fp = hts_file_->fp.bgzf;
    if (bgzf_write(bgzf_fp, record_str.c_str(), record_str.length()) < 0) {
      throw std::runtime_error("BGZF write failed");
    }
  } else {
    hFILE *hfile_fp = hts_file_->fp.hfile;
    // 普通格式，使用hwrite
    if (hwrite(hfile_fp, record_str.c_str(), record_str.length()) < 0) {
      throw std::runtime_error("File write failed");
    }
  }
}

bool BedHandle::has_next() { return ret_code_ >= 0; }

bool BedHandle::next(BedRecord *record) {
  if (hts_itr_) {
    // 使用迭代器读取区间数据
    kstring_t str = {0, 0, NULL};
    ret_code_ = tbx_itr_next(hts_file_.get(), tbx_.get(), hts_itr_.get(), &str);
    if (ret_code_ >= 0) {
      // 解析BED格式行：chr\tstart\tend
      std::string line(str.s, str.l);
      std::istringstream iss(line);
      std::string chrom, start_str, end_str;
      
      if (std::getline(iss, chrom, '\t') && 
          std::getline(iss, start_str, '\t') && 
          std::getline(iss, end_str, '\t')) {
        
        // 查找染色体ID
        int tid = -1;
        for (size_t i = 0; i < chrom_info_.size(); ++i) {
          if (chrom_info_[i].name == chrom) {
            tid = static_cast<int>(i);
            break;
          }
        }
        
        if (tid >= 0) {
          record->tid = tid;
          record->start = std::stoll(start_str);
          record->end = std::stoll(end_str);
        } else {
          ret_code_ = -1; // 未找到染色体
        }
      } else {
        ret_code_ = -1; // 解析失败
      }
    }
    free(str.s);
  } else {
    // 顺序读取整个文件
    kstring_t str = {0, 0, NULL};
    ret_code_ = hts_getline(hts_file_.get(), KS_SEP_LINE, &str);
    if (ret_code_ >= 0) {
      // 解析BED格式行
      std::string line(str.s, str.l);
      std::istringstream iss(line);
      std::string chrom, start_str, end_str;
      
      if (std::getline(iss, chrom, '\t') && 
          std::getline(iss, start_str, '\t') && 
          std::getline(iss, end_str, '\t')) {
        
        // 查找染色体ID
        int tid = -1;
        for (size_t i = 0; i < chrom_info_.size(); ++i) {
          if (chrom_info_[i].name == chrom) {
            tid = static_cast<int>(i);
            break;
          }
        }
        
        if (tid >= 0) {
          record->tid = tid;
          record->start = std::stoll(start_str);
          record->end = std::stoll(end_str);
        } else {
          ret_code_ = -1; // 未找到染色体
        }
      } else {
        ret_code_ = -1; // 解析失败
      }
    }
    free(str.s);
  }
  return ret_code_ >= 0;
}

void BedHandle::close() {
  hts_itr_.reset();
  hts_file_.reset();
}

void BedHandle::query_from_region(const char *region) {
  GenomicIntervalHandle<BedRecord>::query_from_region(region);
  
  
  hts_itr_t *itr = tbx_itr_querys(tbx_.get(), region);
  if (!itr) {
    throw std::runtime_error("Failed to create iterator for region: " + std::string(region));
  }
  
  hts_itr_.reset(itr);
}

void BedHandle::query_from_region(int tid, int64_t start, int64_t end) {
  GenomicIntervalHandle<BedRecord>::query_from_region(tid, start, end);
  
  hts_itr_t *itr = tbx_itr_queryi(tbx_.get(), tid, start, end);
  if (!itr) {
    throw std::runtime_error("Failed to create iterator for region: " + 
                           std::to_string(tid) + ":" + 
                           std::to_string(start) + "-" + 
                           std::to_string(end));
  }
  
  hts_itr_.reset(itr);
}

void BedHandle::query_from_multi_region(char **regions, int num_regions) {
  GenomicIntervalHandle<BedRecord>::query_from_multi_region(regions, num_regions);
    throw std::runtime_error("query_from_multi_region is not supported for BED");
}