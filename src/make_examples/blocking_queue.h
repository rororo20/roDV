#ifndef RODV_MAKE_EXAMPLES_BLOCKING_QUEUE_H_
#define RODV_MAKE_EXAMPLES_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace rodv {
namespace make_examples {

// Bounded producer/consumer queue. Same flow as hc's BlockingQueue: push waits
// while full, pop waits while empty, so a fast stage cannot run ahead of a
// slow one. Timeout pop is omitted; hc's version only compiles for pointer
// types.
template <typename T> class BlockingQueue {
public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity < 1 ? 1 : capacity) {}

  void push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return queue_.size() < capacity_; });
    queue_.push(std::move(value));
    lock.unlock();
    not_empty_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return !queue_.empty(); });
    T value = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return value;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

private:
  size_t capacity_;
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

} // namespace make_examples
} // namespace rodv

#endif
