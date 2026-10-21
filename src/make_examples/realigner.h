#ifndef RODV_MAKE_EXAMPLES_REALIGNER_H_
#define RODV_MAKE_EXAMPLES_REALIGNER_H_

#include "snp_counter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rodv {
namespace make_examples {

// Defaults match DeepVariant r1.6.1 realigner flags. Alignment is global with
// affine gaps, not the striped local Smith-Waterman used upstream. Only
// haplotypes that reach the sink k-mer are kept.
struct RealignerOptions {
  int min_windows_distance = 80;
  int max_window_size = 1000;
  int min_supporting_reads = 2;
  int max_supporting_reads = 300;
  int min_k = 10;
  int max_k = 101;
  int step_k = 1;
  int min_edge_weight = 2;
  int max_num_paths = 256;
  int min_mapq = 14;
  int min_base_quality = 15;
  int match = 4;
  int mismatch = 6;
  int gap_open = 8;
  int gap_extend = 2;
  int max_read_length = 500;
};

struct CigarOp {
  uint32_t op = 0; // BAM_CMATCH, BAM_CINS, or BAM_CDEL
  int length = 0;
};

struct AlignedRead {
  int64_t pos = 0;
  std::vector<uint32_t> cigar;
  bool realigned = false;
};

// Half-open windows. positions are absolute. A cluster of candidates closer
// than 2 * min_windows_distance is one window, padded by min_windows_distance
// and clamped to [0, ref_len).
std::vector<std::pair<int64_t, int64_t>>
assembly_windows(const std::vector<int64_t> &positions, int min_windows_distance,
                 int64_t ref_len);

// de Bruijn haplotypes for one reference window, sorted. Empty when every k
// cycles or the path limit is exceeded; the caller then leaves reads alone.
std::vector<std::string>
assemble_haplotypes(const std::string &ref,
                    const std::vector<const bam1_t *> &reads,
                    const RealignerOptions &options = {});

// Global alignment of query onto target. Cigar is query-relative.
CigarOp align_op(uint32_t op, int length);
std::vector<CigarOp> align_affine(const std::string &query,
                                  const std::string &target,
                                  const RealignerOptions &options = {});

// read_to_hap and hap_to_ref must each cover its haplotype exactly.
std::vector<CigarOp> compose_cigars(const std::vector<CigarOp> &read_to_hap,
                                    const std::vector<CigarOp> &hap_to_ref);

// Realign reads against ref. ref[0] is the base at ref_start. Reads that do
// not overlap an assembly window, or are longer than max_read_length, are
// returned with their original position and cigar.
std::vector<AlignedRead>
realign_reads(const std::string &ref, int64_t ref_start,
              const std::vector<const bam1_t *> &reads,
              const RealignerOptions &options = {});

} // namespace make_examples
} // namespace rodv

#endif
