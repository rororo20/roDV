#include "vcf_handle.h"

void VcfHandle::write_record(bcf1_t *record) {
  if (bcf_write1(hts_file_.get(), header_, record) < 0) {
    throw std::runtime_error("bcf_write1 failed");
  }
}

bool VcfHandle::has_next() { return ret_code_ >= 0; }

bool VcfHandle::next(bcf1_t *record) {
  if (hts_itr_) {
    ret_code_ = bcf_itr_next(hts_file_.get(), hts_itr_.get(), record);
  } else {
    ret_code_ = bcf_read(hts_file_.get(), header_, record);
  }
  return ret_code_ >= 0;
}

void VcfHandle::query_from_region(const char *region) {
  GenomicIntervalHandle<bcf1_t>::query_from_region(region);
  hts_itr_t *itr = tbx_itr_querys(hts_idx_.get(), region);
  hts_itr_.reset(itr);
}

void VcfHandle::query_from_region(int tid, int64_t start, int64_t end) {
  GenomicIntervalHandle<bcf1_t>::query_from_region(tid, start, end);
  hts_itr_t *itr = tbx_itr_queryi(hts_idx_.get(), tid, start, end);
  hts_itr_.reset(itr);
}
