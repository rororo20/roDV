#ifndef RODV_MAKE_EXAMPLES_PILEUP_H_
#define RODV_MAKE_EXAMPLES_PILEUP_H_

#include "runtime/pileup_stage.h"

namespace rodv {
namespace make_examples {

// WGS pileup used by DeepVariant r1.6.1 when alt_aligned_pileup is "none":
// height 100, width 221, 7 channels
// (base, base quality, mapping quality, strand, supports alt, matches ref,
// insert size). Five reference rows sit on top. Pixel values follow
// pileup_image.default_options (base colors, caps, alphas, 254 scale).
class WgsPileup : public PileupStage {
public:
  static constexpr int kHeight = 100;
  static constexpr int kWidth = 221;
  static constexpr int kChannels = 7;
  static constexpr int kOutputChannels = 14;
  static constexpr int kRefRows = 5;

  std::vector<uint8_t>
  encode(int64_t source_id, int64_t ref_start,
         const std::vector<Candidate> &candidates, const std::string &ref,
         const std::vector<bam1_t *> &reads, const std::vector<bam1_t *> &parents,
         bool phase) override;

  // One image, row-major height x width x 14. Channels 0-6 are the
  // ref-aligned pileup. Channels 7-13 are the alt-aligned pileup.
  std::vector<uint8_t>
  encode_one(const Candidate &candidate, int64_t ref_start, const std::string &ref,
             const std::vector<bam1_t *> &reads,
             const std::vector<bam1_t *> &parents = {}, bool phase = false) const;
};

} // namespace make_examples
} // namespace rodv

#endif
