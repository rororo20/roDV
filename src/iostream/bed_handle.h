#include "genomic_interval_handle.h"

#include <string>

class BedHandle : public GenomicIntervalHandle<char *> {
public:
  BedHandle(std::string file_path, ModeType mode)
      : GenomicIntervalHandle<char *>(file_path, mode) {}
};