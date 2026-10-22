#include "runtime/reference_cache.h"

#include <cctype>
#include <stdexcept>

#include <htslib/faidx.h>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rodv {
namespace make_examples {

ReferenceCache::~ReferenceCache() { join(); }

void ReferenceCache::hold_publish() {
  std::lock_guard<std::mutex> lock(mutex_);
  hold_ = true;
}

void ReferenceCache::release_publish() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    hold_ = false;
  }
  cv_.notify_all();
}

void ReferenceCache::start(std::vector<std::string> contigs) {
  join();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = std::move(contigs);
    fasta_path_.clear();
    from_fasta_ = false;
    ready_.clear();
    stop_ = false;
    started_ = true;
  }
  thread_ = std::thread([this] { load(); });
}

void ReferenceCache::start_fasta(const std::string &path) {
  faidx_t *fai = fai_load(path.c_str());
  if (fai == nullptr) {
    throw std::runtime_error("failed to open FASTA: " + path);
  }
  fai_destroy(fai);
  join();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
    fasta_path_ = path;
    from_fasta_ = true;
    ready_.clear();
    stop_ = false;
    started_ = true;
  }
  thread_ = std::thread([this] { load(); });
}

void ReferenceCache::join() {
  if (thread_.joinable()) {
    thread_.join();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  stop_ = true;
  started_ = false;
  cv_.notify_all();
}

std::string ReferenceCache::get(int tid) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return stop_ || ready_.count(tid) != 0; });
  const auto found = ready_.find(tid);
  if (found == ready_.end()) {
    return {};
  }
  return found->second;
}

void ReferenceCache::load() {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "dv-ref");
#endif
  bool from_fasta = false;
  std::string fasta_path;
  std::vector<std::string> contigs;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    from_fasta = from_fasta_;
    fasta_path = fasta_path_;
    contigs = pending_;
  }
  if (from_fasta) {
    load_fasta(fasta_path);
    return;
  }
  for (int tid = 0; tid < static_cast<int>(contigs.size()); ++tid) {
    if (!publish(tid, contigs[static_cast<size_t>(tid)])) {
      return;
    }
  }
}

bool ReferenceCache::publish(int tid, const std::string &bases) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return stop_ || !hold_; });
  if (stop_) {
    return false;
  }
  ready_.emplace(tid, bases);
  lock.unlock();
  cv_.notify_all();
  return true;
}

void ReferenceCache::load_fasta(const std::string &path) {
  faidx_t *fai = fai_load(path.c_str());
  if (fai == nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    cv_.notify_all();
    return;
  }
  const int nseq = faidx_nseq(fai);
  for (int tid = 0; tid < nseq; ++tid) {
    const char *name = faidx_iseq(fai, tid);
    const int seq_len = faidx_seq_len(fai, name);
    int fetched = 0;
    char *raw = seq_len > 0 ? faidx_fetch_seq(fai, name, 0, seq_len - 1, &fetched)
                            : nullptr;
    std::string bases;
    if (raw != nullptr) {
      bases.assign(raw, raw + fetched);
      free(raw);
      for (char &base : bases) {
        base = static_cast<char>(std::toupper(static_cast<unsigned char>(base)));
      }
    }
    if (!publish(tid, bases)) {
      fai_destroy(fai);
      return;
    }
  }
  fai_destroy(fai);
}

} // namespace make_examples
} // namespace rodv
