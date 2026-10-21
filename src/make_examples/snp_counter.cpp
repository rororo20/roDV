#include "snp_counter.h"

#include <algorithm>
#include <string>
#include <vector>

namespace rodv {
namespace make_examples {
namespace {

int base_index(char base) {
  switch (base) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
    return 3;
  default:
    return -1;
  }
}

char index_base(int index) {
  static constexpr char kBases[] = {'A', 'C', 'G', 'T'};
  return kBases[index];
}

enum class EventKind { kRef, kSnp, kInsertion, kDeletion, kSoftClip };

struct Event {
  int64_t pos = 0;
  EventKind kind = EventKind::kRef;
  char snp = 'N';
  std::string bases;
};

bool canonical_base(char base) { return base_index(base) >= 0; }

bool read_bases_usable(const uint8_t *seq, const uint8_t *qual, int read_pos,
                       int len, int min_base_quality) {
  int quality_sum = 0;
  for (int i = 0; i < len; ++i) {
    const char base = seq_nt16_str[bam_seqi(seq, read_pos + i)];
    if (!canonical_base(base)) {
      return false;
    }
    quality_sum += qual[read_pos + i];
  }
  return quality_sum >= min_base_quality * len;
}

void add_indel(LocusCount &locus, bool insertion, const std::string &bases) {
  for (IndelAllele &indel : locus.indels) {
    if (indel.insertion == insertion && indel.bases == bases) {
      ++indel.count;
      return;
    }
  }
  IndelAllele indel;
  indel.insertion = insertion;
  indel.bases = bases;
  indel.count = 1;
  locus.indels.push_back(std::move(indel));
}

} // namespace

void accumulate_snps(const bam1_t *read, int64_t region_start, const char *ref,
                     int64_t region_len, std::vector<LocusCount> &counts,
                     const ReadFilter &filter, int min_base_quality) {
  if (counts.size() < static_cast<size_t>(region_len)) {
    counts.resize(region_len);
  }
  if (!keep_read(read, filter) || region_len <= 0) {
    return;
  }

  const int64_t region_end = region_start + region_len;
  int64_t ref_pos = read->core.pos;
  int read_pos = 0;
  const uint32_t *cigar = bam_get_cigar(read);
  const uint8_t *seq = bam_get_seq(read);
  const uint8_t *qual = bam_get_qual(read);
  std::vector<Event> events;

  auto in_region = [&](int64_t pos) {
    return pos >= region_start && pos < region_end;
  };

  for (uint32_t i = 0; i < read->core.n_cigar; ++i) {
    const uint32_t op = bam_cigar_op(cigar[i]);
    const int len = static_cast<int>(bam_cigar_oplen(cigar[i]));
    const bool is_match =
        op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF;

    if (is_match) {
      for (int j = 0; j < len; ++j) {
        const int64_t pos = ref_pos + j;
        if (!in_region(pos)) {
          continue;
        }
        const int64_t offset = pos - region_start;
        const int ref_base = base_index(ref[offset]);
        const char read_base = seq_nt16_str[bam_seqi(seq, read_pos + j)];
        const int read_index = base_index(read_base);
        if (ref_base < 0 || read_index < 0 ||
            qual[read_pos + j] < min_base_quality) {
          continue;
        }
        Event event;
        event.pos = pos;
        if (read_index == ref_base) {
          event.kind = EventKind::kRef;
        } else {
          event.kind = EventKind::kSnp;
          event.snp = read_base;
        }
        events.push_back(std::move(event));
      }
      ref_pos += len;
      read_pos += len;
      continue;
    }

    if (op == BAM_CINS || op == BAM_CSOFT_CLIP || op == BAM_CDEL) {
      const int64_t anchor = ref_pos - 1;
      char prev = 'N';
      if (read_pos == 0) {
        if (in_region(anchor)) {
          prev = ref[anchor - region_start];
        }
      } else {
        prev = seq_nt16_str[bam_seqi(seq, read_pos - 1)];
      }
      const bool prev_ok = canonical_base(prev) && in_region(anchor);
      if (prev_ok && op != BAM_CDEL &&
          read_bases_usable(seq, qual, read_pos, len, min_base_quality)) {
        Event event;
        event.pos = anchor;
        if (op == BAM_CSOFT_CLIP) {
          event.kind = EventKind::kSoftClip;
        } else {
          event.kind = EventKind::kInsertion;
          event.bases.assign(1, prev);
          for (int j = 0; j < len; ++j) {
            event.bases.push_back(seq_nt16_str[bam_seqi(seq, read_pos + j)]);
          }
        }
        events.push_back(std::move(event));
      } else if (prev_ok && op == BAM_CDEL && ref_pos >= region_start &&
                 ref_pos + len <= region_end) {
        bool deleted_ok = true;
        std::string bases(1, prev);
        for (int j = 0; j < len; ++j) {
          const char deleted = ref[ref_pos - region_start + j];
          if (!canonical_base(deleted)) {
            deleted_ok = false;
            break;
          }
          bases.push_back(deleted);
        }
        if (deleted_ok) {
          Event event;
          event.pos = anchor;
          event.kind = EventKind::kDeletion;
          event.bases = std::move(bases);
          events.push_back(std::move(event));
        }
      }
      if (op == BAM_CDEL) {
        ref_pos += len;
      } else {
        read_pos += len;
      }
      continue;
    }

    if (op == BAM_CREF_SKIP || op == BAM_CPAD) {
      ref_pos += len;
    }
  }

  for (size_t i = 0; i < events.size(); ++i) {
    if (i + 1 < events.size() && events[i].pos == events[i + 1].pos) {
      continue;
    }
    LocusCount &locus = counts[static_cast<size_t>(events[i].pos - region_start)];
    switch (events[i].kind) {
    case EventKind::kRef:
      ++locus.ref;
      break;
    case EventKind::kSnp:
      ++locus.alt[base_index(events[i].snp)];
      break;
    case EventKind::kInsertion:
      add_indel(locus, true, events[i].bases);
      break;
    case EventKind::kDeletion:
      add_indel(locus, false, events[i].bases);
      break;
    case EventKind::kSoftClip:
      ++locus.soft_clip;
      break;
    }
  }
}

std::vector<SnpCandidate>
snp_candidates(const std::vector<LocusCount> &counts, int64_t region_start,
               const char *ref, int min_count, float min_fraction) {
  std::vector<SnpCandidate> found;
  for (size_t i = 0; i < counts.size(); ++i) {
    const int ref_base = base_index(ref[i]);
    if (ref_base < 0) {
      continue;
    }
    const int total = counts[i].total();
    if (total <= 0) {
      continue;
    }
    for (int alt = 0; alt < 4; ++alt) {
      if (alt == ref_base) {
        continue;
      }
      const int alt_count = counts[i].alt[alt];
      if (alt_count < min_count) {
        continue;
      }
      if (static_cast<float>(alt_count) / static_cast<float>(total) <
          min_fraction) {
        continue;
      }
      SnpCandidate candidate;
      candidate.pos = region_start + static_cast<int64_t>(i);
      candidate.ref = index_base(ref_base);
      candidate.alt = index_base(alt);
      candidate.alt_count = alt_count;
      candidate.total = total;
      found.push_back(candidate);
    }
  }
  return found;
}

std::vector<Candidate> candidates(const std::vector<LocusCount> &counts,
                                  int64_t region_start, const char *ref,
                                  const CandidateThresholds &thresholds) {
  std::vector<Candidate> found;
  for (size_t i = 0; i < counts.size(); ++i) {
    const int ref_index = base_index(ref[i]);
    if (ref_index < 0) {
      continue;
    }
    const LocusCount &locus = counts[i];
    const int total = locus.total();
    if (total <= 0) {
      continue;
    }
    const float denom = static_cast<float>(total);

    struct Good {
      AlleleKind kind;
      std::string bases;
      int count;
    };
    std::vector<Good> good;
    for (int alt = 0; alt < 4; ++alt) {
      if (alt == ref_index ||
          locus.alt[alt] < thresholds.snp_min_count) {
        continue;
      }
      if (static_cast<float>(locus.alt[alt]) / denom <
          thresholds.snp_min_fraction) {
        continue;
      }
      good.push_back(
          Good{AlleleKind::kSnp, std::string(1, index_base(alt)), locus.alt[alt]});
    }
    for (const IndelAllele &indel : locus.indels) {
      if (indel.count < thresholds.indel_min_count) {
        continue;
      }
      if (static_cast<float>(indel.count) / denom <
          thresholds.indel_min_fraction) {
        continue;
      }
      good.push_back(Good{indel.insertion ? AlleleKind::kInsertion
                                          : AlleleKind::kDeletion,
                          indel.bases, indel.count});
    }
    if (good.empty()) {
      continue;
    }

    std::string variant_ref(1, index_base(ref_index));
    for (const Good &alt : good) {
      if (alt.kind == AlleleKind::kDeletion &&
          alt.bases.size() > variant_ref.size()) {
        variant_ref = std::string(1, index_base(ref_index)) + alt.bases.substr(1);
      }
    }

    const int64_t pos = region_start + static_cast<int64_t>(i);
    std::vector<Candidate> at_locus;
    for (const Good &alt : good) {
      Candidate candidate;
      candidate.pos = pos;
      candidate.kind = alt.kind;
      candidate.ref = variant_ref;
      candidate.alt_count = alt.count;
      candidate.total = total;
      if (alt.kind == AlleleKind::kDeletion) {
        candidate.alt = alt.bases.substr(0, 1);
        if (alt.bases.size() < variant_ref.size()) {
          candidate.alt += variant_ref.substr(alt.bases.size());
        }
      } else {
        candidate.alt = alt.bases;
        if (variant_ref.size() > 1) {
          candidate.alt += variant_ref.substr(1);
        }
      }
      at_locus.push_back(std::move(candidate));
    }
    std::sort(at_locus.begin(), at_locus.end(),
              [](const Candidate &a, const Candidate &b) { return a.alt < b.alt; });
    found.insert(found.end(), at_locus.begin(), at_locus.end());
  }
  return found;
}

} // namespace make_examples
} // namespace rodv
