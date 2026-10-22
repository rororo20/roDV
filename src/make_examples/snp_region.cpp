#include "snp_region.h"

#include "apply_alignment.h"
#include "region_pipeline.h"

#include <memory>

namespace rodv {
namespace make_examples {

std::vector<RegionSnpResult>
count_snp_regions(const std::vector<SnpRegionInput> &regions, int nthreads,
                  const ReadFilter &filter, int min_base_quality,
                  const CandidateThresholds &thresholds,
                  const RealignerOptions &realigner) {
  std::vector<std::shared_ptr<SnpRegionInput>> jobs;
  jobs.reserve(regions.size());
  for (const SnpRegionInput &region : regions) {
    jobs.push_back(std::make_shared<SnpRegionInput>(region));
  }

  return run_ordered<SnpRegionInput, RegionSnpResult>(
      jobs,
      [=](const SnpRegionInput &region) {
        const std::vector<AlignedRead> realigned =
            realign_reads(region.ref, region.start, region.reads, realigner);
        std::vector<std::unique_ptr<bam1_t, BamDeleter>> owned;
        owned.reserve(region.reads.size());
        std::vector<const bam1_t *> countable;
        countable.reserve(region.reads.size());
        for (size_t i = 0; i < region.reads.size(); ++i) {
          std::unique_ptr<bam1_t, BamDeleter> copy =
              with_alignment(region.reads[i], realigned[i]);
          if (copy) {
            countable.push_back(copy.get());
            owned.push_back(std::move(copy));
          } else {
            countable.push_back(region.reads[i]);
          }
        }

        std::vector<LocusCount> counts;
        for (const bam1_t *read : countable) {
          accumulate_snps(read, region.start, region.ref.data(),
                          static_cast<int64_t>(region.ref.size()), counts,
                          filter, min_base_quality);
        }
        RegionSnpResult result;
        result.start = region.start;
        result.candidates = candidates(counts, region.start, region.ref.data(),
                                       thresholds);
        return result;
      },
      nthreads);
}

} // namespace make_examples
} // namespace rodv

