#ifndef RODV_MAKE_EXAMPLES_REGION_PIPELINE_H_
#define RODV_MAKE_EXAMPLES_REGION_PIPELINE_H_

#include "blocking_queue.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rodv {
namespace make_examples {

// hc uses one dispatch thread, an asio pool of nthreads, and a writer that
// emits by source id. boost::asio is not a dependency here: nthreads workers
// pop a bounded job queue, and one collector emits results in input order.
//
// nthreads is clamped to [1, 128], the same range as hc --nthreads.
// Job-queue and result-queue capacity are nthreads * 2, matching hc's
// n_resource = nthreads * 2.
//
// The caller owns anything Job points at until run_ordered returns. Workers
// must not share a BAM handle; copy reads into the job on the producing side,
// which is what hc's RegionDispatch does.

inline int clamp_nthreads(int nthreads) {
  if (nthreads < 1) {
    return 1;
  }
  if (nthreads > 128) {
    return 128;
  }
  return nthreads;
}

template <typename Job> struct PipelineWork {
  int64_t sequence = -1;
  std::shared_ptr<Job> job;
  bool finish = false;
};

template <typename Result> struct PipelineProduct {
  int64_t sequence = -1;
  Result result;
};

template <typename Job, typename Result>
std::vector<Result>
run_ordered(const std::vector<std::shared_ptr<Job>> &jobs,
            const std::function<Result(const Job &)> &process, int nthreads) {
  std::vector<Result> ordered(jobs.size());
  if (jobs.empty()) {
    return ordered;
  }

  nthreads = clamp_nthreads(nthreads);
  const size_t capacity = static_cast<size_t>(nthreads) * 2;
  BlockingQueue<PipelineWork<Job>> job_queue(capacity);
  BlockingQueue<PipelineProduct<Result>> result_queue(capacity);

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(nthreads));
  for (int i = 0; i < nthreads; ++i) {
    workers.emplace_back([&job_queue, &result_queue, &process] {
#if defined(__linux__)
      pthread_setname_np(pthread_self(), "dv-region");
#endif
      while (true) {
        PipelineWork<Job> work = job_queue.pop();
        if (work.finish) {
          return;
        }
        PipelineProduct<Result> product;
        product.sequence = work.sequence;
        product.result = process(*work.job);
        result_queue.push(std::move(product));
      }
    });
  }

  std::thread collector([&] {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "dv-order");
#endif
    std::vector<PipelineProduct<Result>> pending(jobs.size());
    std::vector<char> ready(jobs.size(), 0);
    int64_t next = 0;
    int64_t got = 0;
    while (got < static_cast<int64_t>(jobs.size())) {
      PipelineProduct<Result> product = result_queue.pop();
      const int64_t sequence = product.sequence;
      pending[static_cast<size_t>(sequence)] = std::move(product);
      ready[static_cast<size_t>(sequence)] = 1;
      ++got;
      while (next < got && ready[static_cast<size_t>(next)]) {
        ordered[static_cast<size_t>(next)] =
            std::move(pending[static_cast<size_t>(next)].result);
        ++next;
      }
    }
  });

  for (size_t i = 0; i < jobs.size(); ++i) {
    PipelineWork<Job> work;
    work.sequence = static_cast<int64_t>(i);
    work.job = jobs[i];
    job_queue.push(std::move(work));
  }
  for (int i = 0; i < nthreads; ++i) {
    PipelineWork<Job> done;
    done.finish = true;
    job_queue.push(std::move(done));
  }

  for (std::thread &worker : workers) {
    worker.join();
  }
  collector.join();
  return ordered;
}

} // namespace make_examples
} // namespace rodv

#endif
