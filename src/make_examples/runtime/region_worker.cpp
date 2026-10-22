#include "runtime/region_worker.h"

#include "apply_alignment.h"
#include "cigar_norm.h"

#include <algorithm>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rodv {
namespace make_examples {

void RegionWorker::run(BlockingQueue<std::shared_ptr<RegionSource>> &jobs,
                       BlockingQueue<ExampleRecord> &results, RegionPool &pool,
                       PileupStage &pileup, const RealignerOptions &realigner,
                       const ReadFilter &filter, int min_base_quality,
                       const CandidateThresholds &thresholds,
                       const std::vector<TruthVariant> &truth, bool gvcf,
                       bool phase) const {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "dv-worker");
#endif
  while (true) {
    std::shared_ptr<RegionSource> job = jobs.pop();
    if (job->finish) {
      if (job->source_id >= 0) {
        ExampleRecord end;
        end.finish = true;
        end.source_id = job->source_id;
        results.push(std::move(end));
      }
      return;
    }

    ExampleRecord record;
    record.source_id = job->source_id;
    record.start = job->start;
    RegionResource &resource = *job->resource;
    for (bam1_t *read : resource.reads) {
      left_align_cigar(read, job->start, job->ref);
    }
    std::vector<const bam1_t *> original(resource.reads.begin(),
                                         resource.reads.end());
    const std::vector<AlignedRead> realigned =
        realign_reads(job->ref, job->start, original, realigner);

    std::vector<BamPtr> owned;
    std::vector<const bam1_t *> countable;
    owned.reserve(original.size());
    countable.reserve(original.size());
    for (size_t i = 0; i < original.size(); ++i) {
      BamPtr copy = with_alignment(original[i], realigned[i]);
      if (copy) {
        countable.push_back(copy.get());
        owned.push_back(std::move(copy));
      } else {
        countable.push_back(original[i]);
      }
    }

    std::vector<LocusCount> counts;
    for (const bam1_t *read : countable) {
      accumulate_snps(read, job->start, job->ref.data(),
                      static_cast<int64_t>(job->ref.size()), counts, filter,
                      min_base_quality);
    }
    record.candidates =
        candidates(counts, job->start, job->ref.data(), thresholds);
    record.labels.assign(record.candidates.size(), -1);
    for (size_t c = 0; c < record.candidates.size(); ++c) {
      for (const TruthVariant &call : truth) {
        if (call.tid == job->tid && call.pos == record.candidates[c].pos &&
            call.alt == record.candidates[c].alt) {
          record.labels[c] = call.genotype;
        }
      }
    }
    if (gvcf) {
      RefBlock block;
      bool open = false;
      for (size_t i = 0; i < counts.size(); ++i) {
        const int64_t pos = job->start + static_cast<int64_t>(i);
        const bool is_candidate = std::any_of(
            record.candidates.begin(), record.candidates.end(),
            [&](const Candidate &candidate) { return candidate.pos == pos; });
        if (!is_candidate && counts[i].total() > 0) {
          if (!open) {
            block.start = pos;
            block.depth = counts[i].total();
            open = true;
          }
          block.end = pos + 1;
          block.depth = std::min(block.depth, counts[i].total());
        } else if (open) {
          record.ref_blocks.push_back(block);
          open = false;
        }
      }
      if (open) {
        record.ref_blocks.push_back(block);
      }
    }
    record.tensors =
        pileup.encode(job->source_id, job->start, record.candidates, job->ref,
                      resource.reads, resource.parent_reads, phase);

    results.push(std::move(record));
    pool.release(std::move(job->resource));
  }
}

} // namespace make_examples
} // namespace rodv
