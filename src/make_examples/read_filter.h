#ifndef RODV_MAKE_EXAMPLES_READ_FILTER_H_
#define RODV_MAKE_EXAMPLES_READ_FILTER_H_

#include <htslib/sam.h>

namespace rodv {
namespace make_examples {

// Defaults match DeepVariant r1.6.1 make_examples:
//   min_mapping_quality=5, drop duplicates, secondary, supplementary, and
//   vendor QC failures. Unmapped reads never overlap a region query.
struct ReadFilter {
  int min_mapping_quality = 5;
  bool keep_duplicates = false;
  bool keep_secondary = false;
  bool keep_supplementary = false;
  bool keep_qcfail = false;
};

inline bool keep_read(const bam1_t *read, const ReadFilter &filter) {
  const uint16_t flag = read->core.flag;
  if (flag & BAM_FUNMAP) {
    return false;
  }
  if ((flag & BAM_FQCFAIL) && !filter.keep_qcfail) {
    return false;
  }
  if ((flag & BAM_FDUP) && !filter.keep_duplicates) {
    return false;
  }
  if ((flag & BAM_FSECONDARY) && !filter.keep_secondary) {
    return false;
  }
  if ((flag & BAM_FSUPPLEMENTARY) && !filter.keep_supplementary) {
    return false;
  }
  return read->core.qual >= filter.min_mapping_quality;
}

} // namespace make_examples
} // namespace rodv

#endif
