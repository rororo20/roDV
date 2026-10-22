#include "apply_alignment.h"

namespace rodv {
namespace make_examples {

BamPtr with_alignment(const bam1_t *src, const AlignedRead &aligned) {
  int query_len = 0;
  for (uint32_t op : aligned.cigar) {
    if (bam_cigar_type(bam_cigar_op(op)) & 1) {
      query_len += static_cast<int>(bam_cigar_oplen(op));
    }
  }
  if (!aligned.realigned || aligned.cigar.empty() ||
      query_len != src->core.l_qseq) {
    return BamPtr(nullptr);
  }

  std::string sequence(static_cast<size_t>(src->core.l_qseq), 'N');
  const uint8_t *raw = bam_get_seq(src);
  for (int i = 0; i < src->core.l_qseq; ++i) {
    sequence[static_cast<size_t>(i)] = seq_nt16_str[bam_seqi(raw, i)];
  }
  BamPtr copy(bam_init1());
  const int rc = bam_set1(
      copy.get(), src->core.l_qname, bam_get_qname(src), src->core.flag,
      src->core.tid, aligned.pos, src->core.qual, aligned.cigar.size(),
      aligned.cigar.data(), src->core.mtid, src->core.mpos, src->core.isize,
      sequence.size(), sequence.data(),
      reinterpret_cast<const char *>(bam_get_qual(src)), 0);
  if (rc < 0) {
    return BamPtr(nullptr);
  }
  return copy;
}

} // namespace make_examples
} // namespace rodv
