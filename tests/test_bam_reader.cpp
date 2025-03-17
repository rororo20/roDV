#include "rodv/bam_reader.h"
#include <gtest/gtest.h>

TEST(BamReaderTest, RegionFiltering) {
    // 准备测试数据（需要实际存在的测试BAM和BED）
    const std::string test_bam = "test/data/region_test.bam";
    const std::string test_bed = "test/data/target_regions.bed";
    
    // 测试BED区域过滤
    BamReader bed_reader(test_bam, test_bed);
    int bed_count = 0;
    bed_reader.LoadAlignments([&](const bam1_t* align) {
        // 验证是否在目标区域
        int pos = align->core.pos + 1; // 0-based转1-based
        EXPECT_GE(pos, 1000);
        EXPECT_LE(pos, 2000);
        bed_count++;
    });
    EXPECT_EQ(bed_count, 42); // 根据实际测试数据调整预期值

    // 测试字符串区域过滤
    BamReader str_reader(test_bam, "", "chr1:1000-2000");
    int str_count = 0;
    str_reader.LoadAlignments([&](const bam1_t* align) {
        str_count++;
    });
    EXPECT_EQ(str_count, bed_count);
}

TEST(BamReaderTest, InvalidInputHandling) {
    // 测试不存在的BAM文件
    EXPECT_THROW(BamReader("invalid.bam"), std::runtime_error);
    
    // 测试无效BED文件
    EXPECT_THROW(BamReader("test.bam", "invalid.bed"), std::runtime_error);
    
    // 测试无效区域字符串
    EXPECT_THROW(BamReader("test.bam", "", "chr1:invalid"), std::runtime_error);
}