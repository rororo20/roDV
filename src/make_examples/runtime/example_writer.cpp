#include "runtime/example_writer.h"

#include "pileup.h"

#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rodv {
namespace make_examples {

void MemorySink::write(const ExampleRecord &record) {
  std::lock_guard<std::mutex> lock(mutex_);
  records_.push_back(record);
}

std::vector<ExampleRecord> MemorySink::records() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return records_;
}

TsvExampleSink::TsvExampleSink(const std::string &path)
    : file_(std::fopen(path.c_str(), "w")),
      pileup_(std::fopen((path + ".pileup").c_str(), "wb")) {
  if (file_ == nullptr || pileup_ == nullptr) {
    throw std::runtime_error("failed to open example output: " + path);
  }
  std::fputs(
      "source_id\tstart\tpos\tkind\tref\talt\talt_count\ttotal\tlabel\tend\n",
      file_);
}

TsvExampleSink::~TsvExampleSink() {
  if (file_ != nullptr) {
    std::fclose(file_);
  }
  if (pileup_ != nullptr) {
    std::fclose(pileup_);
  }
}

void TsvExampleSink::write(const ExampleRecord &record) {
  for (size_t c = 0; c < record.candidates.size(); ++c) {
    const Candidate &candidate = record.candidates[c];
    const char *kind = "SNP";
    if (candidate.kind == AlleleKind::kInsertion) {
      kind = "INS";
    } else if (candidate.kind == AlleleKind::kDeletion) {
      kind = "DEL";
    }
    const int label =
        c < record.labels.size() ? record.labels[c] : -1;
    std::fprintf(file_, "%ld\t%ld\t%ld\t%s\t%s\t%s\t%d\t%d\t%d\t%ld\n",
                 static_cast<long>(record.source_id),
                 static_cast<long>(record.start),
                 static_cast<long>(candidate.pos), kind, candidate.ref.c_str(),
                 candidate.alt.c_str(), candidate.alt_count, candidate.total,
                 label, static_cast<long>(candidate.pos + 1));
  }
  for (const RefBlock &block : record.ref_blocks) {
    std::fprintf(file_, "%ld\t%ld\t%ld\tREF\t.\t.\t%d\t%d\t-1\t%ld\n",
                 static_cast<long>(record.source_id),
                 static_cast<long>(record.start), static_cast<long>(block.start),
                 block.depth, block.depth, static_cast<long>(block.end));
  }
  if (!record.tensors.empty() && pileup_ != nullptr) {
    const int64_t n =
        static_cast<int64_t>(record.tensors.size()) /
        (WgsPileup::kHeight * WgsPileup::kWidth * WgsPileup::kOutputChannels);
    std::fwrite(&record.source_id, sizeof(record.source_id), 1, pileup_);
    std::fwrite(&n, sizeof(n), 1, pileup_);
    std::fwrite(record.tensors.data(), 1, record.tensors.size(), pileup_);
  }
  std::fflush(file_);
}

ExampleWriter::ExampleWriter(BlockingQueue<ExampleRecord> &results,
                             ExampleSink &sink)
    : results_(results), sink_(sink) {}

void ExampleWriter::start() {
  if (thread_.joinable()) {
    return;
  }
  thread_ = std::thread([this] { loop(); });
}

void ExampleWriter::stop() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ExampleWriter::loop() {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "dv-writer");
#endif
  while (true) {
    ExampleRecord record = results_.pop();
    if (record.finish) {
      finish_id_ = record.source_id;
    } else {
      pending_.emplace(record.source_id, std::move(record));
    }
    while (pending_.count(expected_) != 0) {
      sink_.write(pending_.at(expected_));
      pending_.erase(expected_);
      ++expected_;
    }
    if (finish_id_ >= 0 && expected_ == finish_id_) {
      return;
    }
  }
}

} // namespace make_examples
} // namespace rodv
