#include <htslib/hts.h>
#include <htslib/thread_pool.h>
#include <stdexcept>
#include <string>

using namespace std;

enum class QueryType {
  ALL = 0,
  BEDFILE = 1,
  REGION = 2,
};

enum class ModeType {
  READ = 0,
  WRITE = 1,
};

template <typename Record> class GenomicIntervalHandle {
public:
  GenomicIntervalHandle(std::string file_path, ModeType mode)
      : query_type_(QueryType::ALL), mode_(mode) {
    ;
  }

  virtual htsFile *get_htslib_handle() { return nullptr; }

  void set_thread_pool(htsThreadPool *thread_pool) {

    if (get_htslib_handle() == nullptr) {
      throw std::runtime_error("htslib handle is nullptr");
      return;
    }
    hts_set_opt(get_htslib_handle(), HTS_OPT_THREAD_POOL, thread_pool);
  }

  virtual void query_from_bedfile(const char *bedfile) {
    query_type_ = QueryType::BEDFILE;
  };

  virtual void query_from_region(const char *region) {
    query_type_ = QueryType::REGION;
  };

  virtual void query_from_region(int tid, int64_t start, int64_t end) {
    query_type_ = QueryType::REGION;
  };

  ModeType get_mode() { return mode_; }
  QueryType get_query_type() { return query_type_; }
  virtual void write_record(Record *record) { ; };

  bool write(Record *record) {
    if (mode_ == ModeType::WRITE) {
      write_record(record);
      return true;
    } else {
      throw std::runtime_error("mode is not WRITE");
    }
  }

  virtual bool has_next() {}
  virtual void close() {}
  virtual bool next(Record *record) {}

  virtual ~GenomicIntervalHandle() {}

private:
  QueryType query_type_;
  ModeType mode_;
};
