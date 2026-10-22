#ifndef RODV_MAKE_EXAMPLES_REFERENCE_CACHE_H_
#define RODV_MAKE_EXAMPLES_REFERENCE_CACHE_H_

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rodv {
namespace make_examples {

// One background thread publishes contigs in order. get(tid) blocks until
// that contig is published. Does not read BAM.
class ReferenceCache {
public:
  ~ReferenceCache();

  // Loader waits while hold_publish() is set, so tests can observe the block.
  void hold_publish();
  void release_publish();

  void start(std::vector<std::string> contigs);
  // Loads contigs on the background thread via faidx. Throws if the FASTA
  // cannot be opened. Does not read BAM.
  void start_fasta(const std::string &path);
  void join();

  std::string get(int tid);

private:
  void load();
  void load_fasta(const std::string &path);
  bool publish(int tid, const std::string &bases);

  std::vector<std::string> pending_;
  std::string fasta_path_;
  bool from_fasta_ = false;
  std::map<int, std::string> ready_;
  bool hold_ = false;
  bool started_ = false;
  bool stop_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};

} // namespace make_examples
} // namespace rodv

#endif
