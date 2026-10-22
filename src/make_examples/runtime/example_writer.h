#ifndef RODV_MAKE_EXAMPLES_EXAMPLE_WRITER_H_
#define RODV_MAKE_EXAMPLES_EXAMPLE_WRITER_H_

#include "blocking_queue.h"
#include "runtime/region_types.h"

#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace rodv {
namespace make_examples {

class ExampleSink {
public:
  virtual ~ExampleSink() = default;
  virtual void write(const ExampleRecord &record) = 0;
};

class MemorySink : public ExampleSink {
public:
  void write(const ExampleRecord &record) override;
  std::vector<ExampleRecord> records() const;

private:
  mutable std::mutex mutex_;
  std::vector<ExampleRecord> records_;
};

// One candidate per line:
// source_id, start, pos, kind, ref, alt, alt_count, total.
class TsvExampleSink : public ExampleSink {
public:
  explicit TsvExampleSink(const std::string &path);
  ~TsvExampleSink() override;
  void write(const ExampleRecord &record) override;

private:
  std::FILE *file_ = nullptr;
  std::FILE *pileup_ = nullptr;
};

// Writes records in source_id order. Out-of-order completions sit in a map
// until expected_id catches up. Stops when expected_id reaches the finish id.
class ExampleWriter {
public:
  ExampleWriter(BlockingQueue<ExampleRecord> &results, ExampleSink &sink);
  void start();
  void stop();

private:
  void loop();

  BlockingQueue<ExampleRecord> &results_;
  ExampleSink &sink_;
  std::thread thread_;
  int64_t expected_ = 0;
  int64_t finish_id_ = -1;
  std::map<int64_t, ExampleRecord> pending_;
};

} // namespace make_examples
} // namespace rodv

#endif
