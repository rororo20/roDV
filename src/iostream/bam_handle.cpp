#include "bam_handle.h"

void BamHandle::write_record(bam1_t *record) {
  if (sam_write1(hts_file_, header_, record) < 0) {
    throw std::runtime_error("sam_write1 failed");
  }
}

bool BamHandle::has_next() { return ret_code_ >= 0; }

bool BamHandle::next(bam1_t *record) {
  ret_code_ = sam_read1(hts_file_, header_, record);
  return ret_code_ >= 0;
}
