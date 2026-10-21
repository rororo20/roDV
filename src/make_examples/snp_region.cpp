#include "snp_region.h"

#include "region_pipeline.h"

#include <memory>

namespace rodv {
namespace make_examples {
namespace {

struct BamDeleter {
  void operator()(bam1_t *read) const { bam_destroy1(read); }
};

std::unique_ptr<bam1_t, BamDeleter> with_alignment(const bam1_t *src,
                                                   const AlignedRead &aligned) {
  int query_len = 0;
  for (uint32_t op : aligned.cigar) {
    if (bam_cigar_type(bam_cigar_op(op)) & 1) {
      query_len += static_cast<int>(bam_cigar_oplen(op));
    }
  }
  if (!aligned.realigned || aligned.cigar.empty() ||
      query_len != src->core.l_qseq) {
    return std::unique_ptr<bam1_t, BamDeleter>(nullptr);
  }

  std::string sequence(static_cast<size_t>(src->core.l_qseq), 'N');
  const uint8_t *raw = bam_get_seq(src);
  for (int i = 0; i < src->core.l_qseq; ++i) {
    sequence[static_cast<size_t>(i)] = seq_nt16_str[bam_seqi(raw, i)];
  }
  auto copy = std::unique_ptr<bam1_t, BamDeleter>(bam_init1());
  const int rc = bam_set1(
      copy.get(), src->core.l_qname, bam_get_qname(src), src->core.flag,
      src->core.tid, aligned.pos, src->core.qual, aligned.cigar.size(),
      aligned.cigar.data(), src->core.mtid, src->core.mpos, src->core.isize,
      sequence.size(), sequence.data(),
      reinterpret_cast<const char *>(bam_get_qual(src)), 0);
  if (rc < 0) {
    return std::unique_ptr<bam1_t, BamDeleter>(nullptr);
  }
  return copy;
}

} // namespace

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

