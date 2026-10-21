#ifndef RODV_INFERENCE_RUNTIME_H_
#define RODV_INFERENCE_RUNTIME_H_

#include <cstddef>
#include <cstdint>

namespace rodv {
namespace inference {

// DeepVariant r1.6.1 call_variants contract:
//   input  NHWC uint8 pileup, Inception-normalized to float in [-1, 1]
//   output 3 genotype probabilities per example: P(0/0), P(0/1), P(1/1)
//
// The CNN itself is the upstream checkpoint (BSD). This interface is the C++
// runtime we own. OpenVINO / oneDNN is the intended CPU backend on Cascade
// Lake; it is not linked yet. Callers should feed batches produced by
// plan_cpu_inference(), not the upstream default of 1024.
class InferenceRuntime {
public:
  virtual ~InferenceRuntime() = default;

  // images: batch * height * width * channels, NHWC, already normalized.
  // probs:  batch * 3.
  virtual void infer(const float *images, int batch, float *probs) = 0;
};

} // namespace inference
} // namespace rodv

#endif
