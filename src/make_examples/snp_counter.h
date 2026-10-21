#ifndef RODV_MAKE_EXAMPLES_SNP_COUNTER_H_
#define RODV_MAKE_EXAMPLES_SNP_COUNTER_H_

#include "read_filter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rodv {
namespace make_examples {

// One indel allele at a locus. bases is the read's anchor base plus the
// inserted read bases, or plus the deleted reference bases. That is the
// AlleleCounter representation, not the final VCF allele.
struct IndelAllele {
  bool insertion = false;
  std::string bases;
  int count = 0;
};

// Per-locus support inside one region. Index 0 is region_start.
// alt[0..3] is A, C, G, T. Low-quality bases (Phred < min_base_quality) are
// dropped from both the alt count and the total, matching AlleleCounter with
// include_low_quality=false.
//
// A read that has an indel or soft clip anchored at a locus does not also
// vote a match at that locus. Soft clips are counted in total() but are never
// candidates. A deletion whose reference bases run off the end of ref is
// ignored; pad ref if a deletion can cross the region boundary.
struct LocusCount {
  int ref = 0;
  int alt[4] = {};
  int soft_clip = 0;
  std::vector<IndelAllele> indels;

  int total() const {
    int n = ref + soft_clip;
    n += alt[0] + alt[1] + alt[2] + alt[3];
    for (const IndelAllele &indel : indels) {
      n += indel.count;
    }
    return n;
  }
};

struct SnpCandidate {
  int64_t pos = 0; // 0-based, same coordinate system as bam1_t::core.pos
  char ref = 'N';
  char alt = 'N';
  int alt_count = 0;
  int total = 0;
};

enum class AlleleKind { kSnp, kInsertion, kDeletion };

// VCF-style allele. ref may be longer than one base when a deletion is also
// present at the locus, and alt is padded to that same reference.
struct Candidate {
  int64_t pos = 0;
  AlleleKind kind = AlleleKind::kSnp;
  std::string ref;
  std::string alt;
  int alt_count = 0;
  int total = 0;
};

struct CandidateThresholds {
  int snp_min_count = 2;
  float snp_min_fraction = 0.12f;
  int indel_min_count = 2;
  float indel_min_fraction = 0.06f;
};

// ref[i] is the reference base at region_start + i, length region_len.
// counts is resized to region_len and accumulated into, not cleared.
void accumulate_snps(const bam1_t *read, int64_t region_start, const char *ref,
                     int64_t region_len, std::vector<LocusCount> &counts,
                     const ReadFilter &filter = {}, int min_base_quality = 10);

// SNP alleles only. r1.6.1 defaults: min_count=2, min_fraction=0.12.
std::vector<SnpCandidate>
snp_candidates(const std::vector<LocusCount> &counts, int64_t region_start,
               const char *ref, int min_count = 2, float min_fraction = 0.12f);

// SNP and indel alleles. Indel defaults are min_count=2, min_fraction=0.06.
std::vector<Candidate>
candidates(const std::vector<LocusCount> &counts, int64_t region_start,
           const char *ref, const CandidateThresholds &thresholds = {});

} // namespace make_examples
} // namespace rodv

#endif
