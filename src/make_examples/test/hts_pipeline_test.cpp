#include "runtime/example_writer.h"
#include "runtime/hts_read_source.h"
#include "runtime/pipeline.h"
#include "runtime/reference_cache.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include <htslib/sam.h>

namespace {

std::string write_fasta(const std::string &path) {
  std::ofstream out(path);
  out << ">chr1\nACGTACGTACGT\n";
  return path;
}

void write_bam(const std::string &path) {
  sam_hdr_t *header = sam_hdr_init();
  ASSERT_EQ(sam_hdr_add_line(header, "SQ", "SN", "chr1", "LN", "12", nullptr), 0);
  htsFile *file = hts_open(path.c_str(), "wb");
  ASSERT_NE(file, nullptr);
  ASSERT_EQ(sam_hdr_write(file, header), 0);
  const uint32_t cigar = bam_cigar_gen(8, BAM_CMATCH);
  const char *qual = ">>>>>>>>";
  for (int i = 0; i < 2; ++i) {
    bam1_t *read = bam_init1();
    ASSERT_GE(bam_set1(read, 1, "r", 0, 0, 0, 60, 1, &cigar, -1, -1, 0, 8,
                       "ACGAACGT", qual, 0),
              0);
    ASSERT_GE(sam_write1(file, header, read), 0);
    bam_destroy1(read);
  }
  ASSERT_EQ(hts_close(file), 0);
  sam_hdr_destroy(header);
}

} // namespace

TEST(HtsPipelineTest, BamAndFastaProduceTheSnpCandidate) {
  const std::string dir = "/tmp/rodv-hts-pipeline";
  const std::string fasta = write_fasta(dir + ".fa");
  const std::string bam = dir + ".bam";
  const std::string out = dir + ".tsv";
  write_bam(bam);

  rodv::make_examples::ReferenceCache cache;
  cache.start_fasta(fasta);
  rodv::make_examples::HtsReadSource source(bam, /*decode_threads=*/2, fasta);
  rodv::make_examples::MemorySink sink;
  rodv::make_examples::PipelineConfig config;
  config.nthreads = 2;
  config.region_bp = 1000;
  rodv::make_examples::MakeExamplesPipeline pipeline(config, nullptr);
  pipeline.run(source, cache, sink);
  cache.join();

  const auto records = sink.records();
  ASSERT_EQ(records.size(), 1u);
  ASSERT_EQ(records[0].candidates.size(), 1u);
  EXPECT_EQ(records[0].candidates[0].pos, 3);
  EXPECT_EQ(records[0].candidates[0].ref, "T");
  EXPECT_EQ(records[0].candidates[0].alt, "A");

  rodv::make_examples::TsvExampleSink tsv(out);
  tsv.write(records[0]);
}
