#include "rodv/bam_reader.h"
#include <htslib/sam.h>
#include <vector> // 假设新增的头文件为vector，可根据实际需求替换

void BamReader::AddRegionsFromBed(const std::string& bed_path) {
    htsFile* bed = hts_open(bed_path.c_str(), "r");
    if (!bed) throw std::runtime_error("Failed to open BED file");
    
    kstring_t line = {0};
    while (hts_getline(bed, '\n', &line) > 0) {
        // 跳过注释行和空行
        if (line.l == 0 || line.s[0] == '#') continue;
        
        // 分割制表符
        std::vector<char*> tokens;
        char* p = strtok(line.s, "\t");
        while (p && tokens.size() < 3) {
            tokens.push_back(p);
            p = strtok(nullptr, "\t");
        }
        
        // 验证BED格式
        if (tokens.size() < 3) {
            hts_close(bed);
            throw std::runtime_error("Invalid BED format in line: " + std::string(line.s));
        }
    
        // 转换坐标（BED是0-based，转换为1-based）
        try {
            const std::string chrom = tokens[0];
            int start = std::stoi(tokens[1]) + 1;
            int end = std::stoi(tokens[2]);
    
            // 检查染色体是否存在
            if (contig_lengths_.find(chrom) == contig_lengths_.end()) {
                throw std::runtime_error("Unknown contig: " + chrom);
            }
            
            // 检查坐标有效性
            const int max_pos = contig_lengths_.at(chrom);
            if (start <= 0 || end > max_pos || start > end) {
                throw std::runtime_error("Invalid coordinates for " + chrom 
                                       + ": " + std::to_string(start) + "-" + std::to_string(end));
            }
            // 判断bed的start和end是否有效
            if (start <= 0 || end <= 0 || start > end) {
                hts_close(bed);
                throw std::runtime_error("Invalid coordinates in BED line: " + std::string(line.s));
            }
            // 判断是否超过reference的长度，如果超过则报错  
            if (end > reference_length_) {
                hts_close(bed);
                throw std::runtime_error("Invalid coordinates in BED line: " + std::string(line.s));    
            }
            
            
            // 构建region字符串格式：chr:start-end
            regions_.emplace_back(std::string(tokens[0]) + ":" 
                                + std::to_string(start) + "-"
                                + std::to_string(end));
        } catch (...) {
            hts_close(bed);
            throw std::runtime_error("Invalid coordinates in BED line: " + std::string(line.s));
        }
    }
    
    hts_close(bed);
    if (line.s) free(line.s);
    
    if (regions_.empty()) {
        throw std::runtime_error("No valid regions found in BED file");
    }
}

void BamReader::LoadAlignments(const std::string& bam_path) {
    htsFile* fp = sam_open(bam_path.c_str(), "r");
    if (!fp) throw std::runtime_error("Failed to open BAM file");
    
    bam_hdr_t* header = sam_hdr_read(fp);
    bam1_t* alignment = bam_init1();
    
    while (sam_read1(fp, header, alignment) >= 0) {
        // 高性能C++解析实现
        ProcessAlignment(alignment); 
    }
    
    // ... 内存清理代码 ...
}

// 补全AddRegionsFromString实现
void BamReader::AddRegionsFromString(const std::string& regions_str) {
    std::istringstream iss(regions_str);
    std::string region;
    while (std::getline(iss, region, ',')) {
        // 去除首尾空格
        region.erase(0, region.find_first_not_of(" "));
        region.erase(region.find_last_not_of(" ") + 1);
        
        // 验证区域格式（chr:start-end）
        if (region.find(":") == std::string::npos || 
            region.find("-") == std::string::npos) {
            throw std::runtime_error("Invalid region format: " + region);
        }
        regions_.push_back(region);
    }
}

BamHandle::BamHandle(const std::string& path) {
    fp_ = sam_open(path.c_str(), "r");
    if (!fp_) throw std::runtime_error("Failed to open BAM: " + path);
    
    header_ = sam_hdr_read(fp_);
    if (!header_ || header_->n_targets == 0) {
        sam_close(fp_);
        throw std::runtime_error("Invalid BAM header");
    }
}

BamHandle::~BamHandle() {
    if (header_) bam_hdr_destroy(header_);
    if (fp_) sam_close(fp_);
}

// 修改后的构造函数
BamReader::BamReader(const std::string& bam_path,
                     const std::string& bed_path,
                     const std::string& regions_str)
    : bam_handle_(std::make_unique<BamHandle>(bam_path)),
      idx_(sam_index_load(bam_handle_->fp_, bam_path.c_str()), hts_idx_destroy)  // 加载BAM索引
{
    // 加载所有contig长度
    bam_hdr_t* header = bam_handle_->header();
    for (int i = 0; i < header->n_targets; ++i) {
        contig_lengths_[header->target_name[i]] = header->target_len[i];
    }

    // 区域解析
    if (!bed_path.empty()) AddRegionsFromBed(bed_path);
    if (!regions_str.empty()) AddRegionsFromString(regions_str);
}

// 新增contig长度访问方法
int BamReader::contig_length(const std::string& contig) const {
    auto it = contig_lengths_.find(contig);
    if (it == contig_lengths_.end()) {
        throw std::out_of_range("Contig not found: " + contig);
    }
    return it->second;
}

bool BamReader::NextAlignment(bam1_t* alignment) {
    // 使用智能指针管理的迭代器进行读取
    // 统一处理两种读取模式
    int ret = sam_itr_next(bam_handle_->fp_, iterator_.get(), alignment);
    
    // 错误处理（适用于两种模式）
    if (ret < -1) {
        throw std::runtime_error("BAM读取错误: " + std::to_string(ret));
    }
    return ret >= 0;  // >=0表示成功，-1表示结束
}