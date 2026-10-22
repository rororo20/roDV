#include "pileup.h"

#include <algorithm>
#include <cmath>

namespace rodv {
namespace make_examples {
namespace {

constexpr float kMaxPixel = 254.0f;

int base_color(char base) {
  switch (base) {
  case 'A':
    return 40 + 70 * 3;
  case 'G':
    return 40 + 70 * 2;
  case 'T':
    return 30 + 70;
  case 'C':
    return 30;
  default:
    return 0;
  }
}

uint8_t scale_color(int value, int cap) {
  if (value > cap) {
    value = cap;
  }
  if (value < 0) {
    value = 0;
  }
  return static_cast<uint8_t>(kMaxPixel * (static_cast<float>(value) / cap));
}

void paint(std::vector<uint8_t> &image, int row, int col, char base, int bq,
           int mq, bool forward, bool supports_alt, bool matches_ref,
           int insert_size) {
  if (row < 0 || col < 0 || row >= WgsPileup::kHeight || col >= WgsPileup::kWidth) {
    return;
  }
  uint8_t *pixel =
      image.data() + (static_cast<size_t>(row) * WgsPileup::kWidth + col) *
                         WgsPileup::kChannels;
  pixel[0] = static_cast<uint8_t>(base_color(base));
  pixel[1] = scale_color(bq, 40);
  pixel[2] = scale_color(mq, 60);
  pixel[3] = forward ? 70 : 240;
  pixel[4] = static_cast<uint8_t>(kMaxPixel * (supports_alt ? 1.0f : 0.6f));
  pixel[5] = static_cast<uint8_t>(kMaxPixel * (matches_ref ? 0.2f : 1.0f));
  int fragment = std::abs(insert_size);
  if (fragment > 1000) {
    fragment = 1000;
  }
  pixel[6] = scale_color(fragment, 1000);
}

char ref_base_at(const std::string &ref, int64_t ref_start, int64_t pos) {
  const int64_t offset = pos - ref_start;
  if (offset < 0 || offset >= static_cast<int64_t>(ref.size())) {
    return 'N';
  }
  return ref[static_cast<size_t>(offset)];
}

bool supports(const Candidate &candidate, char read_base, bool indel_here) {
  if (candidate.kind == AlleleKind::kSnp) {
    return !candidate.alt.empty() && read_base == candidate.alt[0];
  }
  return indel_here;
}

} // namespace

std::vector<uint8_t>
WgsPileup::encode_one(const Candidate &candidate, int64_t ref_start,
                      const std::string &ref, const std::vector<bam1_t *> &reads,
                      const std::vector<bam1_t *> &parents, bool phase) const {
  auto window_base = [&](int col) {
    return ref_base_at(ref, ref_start, candidate.pos - ((kWidth - 1) / 2) + col);
  };
  std::string alt_window(static_cast<size_t>(kWidth), 'N');
  for (int col = 0; col < kWidth; ++col) {
    alt_window[static_cast<size_t>(col)] = window_base(col);
  }
  const int center = (kWidth - 1) / 2;
  if (candidate.kind == AlleleKind::kSnp && !candidate.alt.empty()) {
    alt_window[static_cast<size_t>(center)] = candidate.alt[0];
  } else if (candidate.kind == AlleleKind::kInsertion && candidate.alt.size() > 1) {
    alt_window.insert(static_cast<size_t>(center) + 1, candidate.alt.substr(1));
    alt_window.resize(static_cast<size_t>(kWidth), 'N');
  } else if (candidate.kind == AlleleKind::kDeletion &&
             candidate.ref.size() > 1) {
    const size_t drop = candidate.ref.size() - 1;
    const size_t at = static_cast<size_t>(center) + 1;
    if (at < alt_window.size()) {
      alt_window.erase(at, std::min(drop, alt_window.size() - at));
      alt_window.resize(static_cast<size_t>(kWidth), 'N');
    }
  }

  auto paint_view = [&](std::vector<uint8_t> &image, const std::string &view,
                        const std::vector<bam1_t *> &sample, int row_begin,
                        int row_end) {
    for (int col = 0; col < kWidth; ++col) {
      for (int row = 0; row < kRefRows; ++row) {
        paint(image, row, col, view[static_cast<size_t>(col)], 60, 60, true, false,
              true, 0);
      }
    }
    std::vector<bam1_t *> ordered = sample;
    const int64_t image_start = candidate.pos - center;
    std::sort(ordered.begin(), ordered.end(), [&](bam1_t *a, bam1_t *b) {
      if (!phase || candidate.kind != AlleleKind::kSnp) {
        return a->core.pos < b->core.pos;
      }
      const auto hp = [&](bam1_t *read) {
        if (read->core.pos > candidate.pos ||
            read->core.pos + read->core.l_qseq <= candidate.pos) {
          return 3;
        }
        const int q = static_cast<int>(candidate.pos - read->core.pos);
        const char base = seq_nt16_str[bam_seqi(bam_get_seq(read), q)];
        if (!candidate.alt.empty() && base == candidate.alt[0]) {
          return 1;
        }
        if (base == view[static_cast<size_t>(center)]) {
          return 2;
        }
        return 3;
      };
      const int ha = hp(a);
      const int hb = hp(b);
      if (ha != hb) {
        return ha < hb;
      }
      return a->core.pos < b->core.pos;
    });
    int row = row_begin;
    for (bam1_t *read : ordered) {
      if (row >= row_end) {
        break;
      }
      const bool forward = (read->core.flag & BAM_FREVERSE) == 0;
      const uint32_t *cigar = bam_get_cigar(read);
      const uint8_t *seq = bam_get_seq(read);
      const uint8_t *qual = bam_get_qual(read);
      int64_t ref_pos = read->core.pos;
      int read_pos = 0;
      bool drew = false;
      bool low_at_call = false;
      for (uint32_t c = 0; c < read->core.n_cigar && !low_at_call; ++c) {
        const uint32_t op = bam_cigar_op(cigar[c]);
        const int len = static_cast<int>(bam_cigar_oplen(cigar[c]));
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
          for (int j = 0; j < len; ++j) {
            const int64_t pos = ref_pos + j;
            const int col = static_cast<int>(pos - image_start);
            if (col < 0 || col >= kWidth) {
              continue;
            }
            const char base = seq_nt16_str[bam_seqi(seq, read_pos + j)];
            const int bq = qual[read_pos + j];
            if (pos == candidate.pos && bq < 10) {
              low_at_call = true;
              break;
            }
            paint(image, row, col, base, bq, read->core.qual, forward,
                  supports(candidate, base, false),
                  base == view[static_cast<size_t>(col)], read->core.isize);
            drew = true;
          }
          ref_pos += len;
          read_pos += len;
        } else if (op == BAM_CINS || op == BAM_CDEL) {
          const int col = static_cast<int>((ref_pos - 1) - image_start);
          if (col >= 0 && col < kWidth) {
            paint(image, row, col, '*', 30, read->core.qual, forward,
                  supports(candidate, '*', true), false, read->core.isize);
            drew = true;
          }
          if (op == BAM_CDEL) {
            ref_pos += len;
          } else {
            read_pos += len;
          }
        } else if (op == BAM_CSOFT_CLIP) {
          read_pos += len;
        } else if (op == BAM_CREF_SKIP || op == BAM_CPAD) {
          ref_pos += len;
        }
      }
      if (drew && !low_at_call) {
        ++row;
      }
    }
  };

  std::string ref_window(static_cast<size_t>(kWidth), 'N');
  for (int col = 0; col < kWidth; ++col) {
    ref_window[static_cast<size_t>(col)] = window_base(col);
  }
  std::vector<uint8_t> ref_image(static_cast<size_t>(kHeight) * kWidth * kChannels,
                                 0);
  std::vector<uint8_t> alt_image(ref_image.size(), 0);
  const int child_end =
      parents.empty() ? kHeight : kRefRows + (kHeight - kRefRows) / 2;
  paint_view(ref_image, ref_window, reads, kRefRows, child_end);
  paint_view(alt_image, alt_window, reads, kRefRows, child_end);
  if (!parents.empty()) {
    paint_view(ref_image, ref_window, parents, child_end, kHeight);
    paint_view(alt_image, alt_window, parents, child_end, kHeight);
  }

  std::vector<uint8_t> image(static_cast<size_t>(kHeight) * kWidth * kOutputChannels,
                             0);
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      const size_t src =
          (static_cast<size_t>(row) * kWidth + col) * kChannels;
      uint8_t *dst = image.data() +
                     (static_cast<size_t>(row) * kWidth + col) * kOutputChannels;
      std::copy(ref_image.begin() + static_cast<std::ptrdiff_t>(src),
                ref_image.begin() + static_cast<std::ptrdiff_t>(src + kChannels),
                dst);
      std::copy(alt_image.begin() + static_cast<std::ptrdiff_t>(src),
                alt_image.begin() + static_cast<std::ptrdiff_t>(src + kChannels),
                dst + kChannels);
    }
  }
  return image;
}

std::vector<uint8_t>
WgsPileup::encode(int64_t source_id, int64_t ref_start,
                  const std::vector<Candidate> &candidates, const std::string &ref,
                  const std::vector<bam1_t *> &reads,
                  const std::vector<bam1_t *> &parents, bool phase) {
  (void)source_id;
  std::vector<uint8_t> packed;
  const size_t one =
      static_cast<size_t>(kHeight) * kWidth * kOutputChannels;
  packed.reserve(one * candidates.size());
  for (const Candidate &candidate : candidates) {
    std::vector<uint8_t> image =
        encode_one(candidate, ref_start, ref, reads, parents, phase);
    packed.insert(packed.end(), image.begin(), image.end());
  }
  return packed;
}

} // namespace make_examples
} // namespace rodv
