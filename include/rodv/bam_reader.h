#ifndef RODV_BAM_READER_H
#define RODV_BAM_READER_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <htslib/sam.h>

namespace rodv {

class BamHandle {
public:
    explicit BamHandle(const std::string& path);
    ~BamHandle();
    
    bam_hdr_t* header() const { return header_; }
    operator htsFile*() const { return fp_; }
    
private:
    htsFile* fp_ = nullptr;
    bam_hdr_t* header_ = nullptr;
};

class BamReader {
public:
    // 修改构造函数，接收bam_path
    explicit BamReader(const std::string& bam_path,
                     const std::string& bed_path = "",
                     const std::string& regions_str = "");
    
    // 新增迭代接口
    bool NextAlignment(bam1_t* alignment);
    // 成员函数声明
    void AddRegionsFromBed(const std::string& bed_path);
    void AddRegionsFromString(const std::string& regions_str);
    int contig_length(const std::string& contig) const;
    
private:
    // 资源管理成员
    std::unique_ptr<BamHandle> bam_handle_;
    std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> idx_{nullptr, hts_idx_destroy};
    std::unique_ptr<hts_itr_t, decltype(&hts_itr_destroy)> iterator_{nullptr, hts_itr_destroy};

    // 数据成员
    std::string bam_path_;
    std::vector<std::string> regions_;
    std::unordered_map<std::string, int> contig_lengths_;
};

} // namespace rodv

#endif // RODV_BAM_READER_H