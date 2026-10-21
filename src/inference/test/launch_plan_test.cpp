#include "launch_plan.h"

#include <gtest/gtest.h>

namespace {

rodv::inference::CpuBudget xeon_8260() {
  rodv::inference::CpuBudget cpu;
  cpu.numa_nodes = 2;
  cpu.physical_cores_per_node = 24;
  cpu.logical_cpus_per_node = 48;
  cpu.l3_bytes_per_node = 36608ull * 1024;
  return cpu;
}

} // namespace

TEST(LaunchPlanTest, DualSocket8260DoesNotUseUpstreamBatch) {
  const rodv::inference::ImageShape wgs;
  const auto plan = rodv::inference::plan_cpu_inference(xeon_8260(), wgs);

  EXPECT_EQ(plan.processes, 8);
  EXPECT_EQ(plan.streams_per_node, 4);
  EXPECT_EQ(plan.threads_per_process, 6);
  EXPECT_EQ(plan.batch_size, 8);
  EXPECT_LT(plan.batch_size, plan.upstream_batch_size);

  const std::string text =
      rodv::inference::format_launch_plan(xeon_8260(), wgs, plan);
  EXPECT_NE(text.find("batch size: 8"), std::string::npos);
  EXPECT_NE(text.find("OpenVINO"), std::string::npos);
}

TEST(LaunchPlanTest, SmallCpuIsOneStream) {
  rodv::inference::CpuBudget cpu;
  cpu.numa_nodes = 1;
  cpu.physical_cores_per_node = 8;
  cpu.logical_cpus_per_node = 8;
  cpu.l3_bytes_per_node = 16ull << 20;

  const auto plan =
      rodv::inference::plan_cpu_inference(cpu, rodv::inference::ImageShape{});
  EXPECT_EQ(plan.processes, 1);
  EXPECT_EQ(plan.threads_per_process, 8);
  EXPECT_GE(plan.batch_size, 8);
  EXPECT_LE(plan.batch_size, 32);
}

TEST(LaunchPlanTest, DetectedBudgetIsUsable) {
  const auto cpu = rodv::inference::detect_cpu_budget();
  EXPECT_GE(cpu.numa_nodes, 1);
  EXPECT_GE(cpu.physical_cores_per_node, 1);
  EXPECT_GE(cpu.logical_cpus_per_node, cpu.physical_cores_per_node);
  EXPECT_GT(cpu.l3_bytes_per_node, 0u);

  const auto plan =
      rodv::inference::plan_cpu_inference(cpu, rodv::inference::ImageShape{});
  EXPECT_GE(plan.processes, cpu.numa_nodes);
  EXPECT_GE(plan.threads_per_process, 1);
  EXPECT_GE(plan.batch_size, 8);
  EXPECT_LE(plan.batch_size, 32);
}
