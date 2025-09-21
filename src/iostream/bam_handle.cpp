#include "bam_handle.h"

void BamHandle::write_record(bam1_t *record) {
  if (sam_write1(hts_file_.get(), header_, record) < 0) {
    throw std::runtime_error("sam_write1 failed");
  }
}

bool BamHandle::has_next() { return ret_code_ >= 0; }

bool BamHandle::next(bam1_t *record) {
  if (hts_itr_) {
    // 使用迭代器读取区间数据
    if (get_query_type() == QueryType::REGION) {
      ret_code_ = sam_itr_next(hts_file_.get(), hts_itr_.get(), record);
    } else if (get_query_type() == QueryType::BEDFILE) {
      ret_code_ = sam_itr_multi_next(hts_file_.get(), hts_itr_.get(), record);
    } else {
      throw std::runtime_error("Invalid query type");
    }
  } else {
    // 顺序读取整个文件
    ret_code_ = sam_read1(hts_file_.get(), header_, record);
  }
  return ret_code_ >= 0;
}

void BamHandle::load_index(const std::string &index_path) {
  if (get_mode() != ModeType::READ) {
    throw std::runtime_error("Index loading is only supported in READ mode");
  }
  hts_idx_t *idx = nullptr;
  if (index_path.empty()) {
    // 尝试自动加载索引文件
    idx = sam_index_load(hts_file_.get(), nullptr);
  } else {
    idx = sam_index_load2(hts_file_.get(), index_path.c_str(), nullptr);
  }
  hts_idx_.reset(idx);
}

void BamHandle::query_from_region(const char *region) {
  GenomicIntervalHandle<bam1_t>::query_from_region(region);
  hts_itr_t *itr = sam_itr_querys(hts_idx_.get(), header_, region);
  if (!itr) {
    throw std::runtime_error("Failed to create iterator for region: " +
                             std::string(region));
  }
  hts_itr_.reset(itr);
}

void BamHandle::query_from_region(int tid, int64_t start, int64_t end) {
  GenomicIntervalHandle<bam1_t>::query_from_region(tid, start, end);

  hts_itr_t *itr = sam_itr_queryi(hts_idx_.get(), tid, start, end);
  if (!itr) {
    throw std::runtime_error(
        "Failed to create iterator for region: " + std::to_string(tid) + ":" +
        std::to_string(start) + "-" + std::to_string(end));
  }

  hts_itr_.reset(itr);
}

void BamHandle::query_from_multi_region(char **regions, int num_regions) {
  GenomicIntervalHandle<bam1_t>::query_from_multi_region(regions, num_regions);

  hts_itr_t *itr = sam_itr_regarray(hts_idx_.get(), header_, regions, 0);

  hts_itr_.reset(itr);
}
