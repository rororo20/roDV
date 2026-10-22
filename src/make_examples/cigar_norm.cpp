#include "cigar_norm.h"

#include <string>
#include <vector>

namespace rodv {
namespace make_examples {
namespace {

struct Op {
  uint32_t code = BAM_CMATCH;
  int len = 0;
};

char ref_at(const std::string &ref, int64_t ref_start, int64_t pos) {
  const int64_t offset = pos - ref_start;
  if (offset < 0 || offset >= static_cast<int64_t>(ref.size())) {
    return 'N';
  }
  return ref[static_cast<size_t>(offset)];
}

std::string sequence_of(const bam1_t *read) {
  std::string seq(static_cast<size_t>(read->core.l_qseq), 'N');
  const uint8_t *raw = bam_get_seq(read);
  for (int i = 0; i < read->core.l_qseq; ++i) {
    seq[static_cast<size_t>(i)] = seq_nt16_str[bam_seqi(raw, i)];
  }
  return seq;
}

std::vector<Op> parse_cigar(const bam1_t *read) {
  std::vector<Op> ops;
  const uint32_t *cigar = bam_get_cigar(read);
  ops.reserve(read->core.n_cigar);
  for (uint32_t i = 0; i < read->core.n_cigar; ++i) {
    ops.push_back(Op{bam_cigar_op(cigar[i]),
                     static_cast<int>(bam_cigar_oplen(cigar[i]))});
  }
  return ops;
}

void collapse(std::vector<Op> &ops) {
  std::vector<Op> merged;
  for (const Op &op : ops) {
    if (op.len <= 0) {
      continue;
    }
    if (!merged.empty() && merged.back().code == op.code) {
      merged.back().len += op.len;
    } else {
      merged.push_back(op);
    }
  }
  ops.swap(merged);
}

} // namespace

bool left_align_cigar(bam1_t *read, int64_t ref_start, const std::string &ref) {
  std::vector<Op> ops = parse_cigar(read);
  const std::string seq = sequence_of(read);
  bool changed = false;
  int64_t ref_pos = read->core.pos;
  int read_pos = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    const uint32_t code = ops[i].code;
    const int len = ops[i].len;
    if (code == BAM_CDEL && i > 0 && ops[i - 1].code == BAM_CMATCH &&
        i + 1 < ops.size() && ops[i + 1].code == BAM_CMATCH) {
      while (ops[i - 1].len > 0 && ref_pos > ref_start &&
             ref_at(ref, ref_start, ref_pos - 1) ==
                 ref_at(ref, ref_start, ref_pos + ops[i].len - 1) &&
             ref_at(ref, ref_start, ref_pos - 1) != 'N') {
        --ops[i - 1].len;
        ++ops[i + 1].len;
        --ref_pos;
        changed = true;
      }
      ref_pos += ops[i].len;
    } else if (code == BAM_CINS && i > 0 && ops[i - 1].code == BAM_CMATCH &&
               read_pos > 0 && len > 0) {
      while (ops[i - 1].len > 0 &&
             seq[static_cast<size_t>(read_pos - 1)] ==
                 seq[static_cast<size_t>(read_pos + ops[i].len - 1)] &&
             seq[static_cast<size_t>(read_pos - 1)] != 'N') {
        --ops[i - 1].len;
        if (i + 1 < ops.size() && ops[i + 1].code == BAM_CMATCH) {
          ++ops[i + 1].len;
        } else {
          ops.insert(ops.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                     Op{BAM_CMATCH, 1});
        }
        --read_pos;
        changed = true;
      }
      read_pos += ops[i].len;
    } else {
      if (bam_cigar_type(code) & 2) {
        ref_pos += len;
      }
      if (bam_cigar_type(code) & 1) {
        read_pos += len;
      }
    }
  }
  if (!changed) {
    return false;
  }
  collapse(ops);
  std::vector<uint32_t> packed;
  packed.reserve(ops.size());
  for (const Op &op : ops) {
    packed.push_back(bam_cigar_gen(static_cast<uint32_t>(op.len), op.code));
  }
  const std::string qual(static_cast<size_t>(read->core.l_qseq), '\0');
  const uint8_t *raw_qual = bam_get_qual(read);
  std::string quality(reinterpret_cast<const char *>(raw_qual),
                      static_cast<size_t>(read->core.l_qseq));
  bam_set1(read, read->core.l_qname, bam_get_qname(read), read->core.flag,
           read->core.tid, read->core.pos, read->core.qual, packed.size(),
           packed.data(), read->core.mtid, read->core.mpos, read->core.isize,
           seq.size(), seq.data(), quality.data(), 0);
  return true;
}

} // namespace make_examples
} // namespace rodv
