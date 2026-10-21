#include "realigner.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace rodv {
namespace make_examples {
namespace {

constexpr int kNeg = -1000000000;

struct Graph {
  struct Edge {
    int to = 0;
    int weight = 0;
    bool is_ref = false;
  };
  std::vector<std::string> kmers;
  std::unordered_map<std::string, int> index;
  std::vector<std::vector<Edge>> out;
  int source = -1;
  int sink = -1;

  int node(const std::string &kmer) {
    const auto found = index.find(kmer);
    if (found != index.end()) {
      return found->second;
    }
    const int id = static_cast<int>(kmers.size());
    index.emplace(kmer, id);
    kmers.push_back(kmer);
    out.emplace_back();
    return id;
  }

  void add_edge(int from, int to, bool is_ref) {
    for (Edge &edge : out[static_cast<size_t>(from)]) {
      if (edge.to == to) {
        ++edge.weight;
        edge.is_ref = edge.is_ref || is_ref;
        return;
      }
    }
    out[static_cast<size_t>(from)].push_back(Edge{to, 1, is_ref});
  }
};

bool ref_kmer_unique(const std::string &ref, int k) {
  std::unordered_set<std::string> seen;
  for (int i = 0; i + k <= static_cast<int>(ref.size()); ++i) {
    if (!seen.insert(ref.substr(static_cast<size_t>(i), static_cast<size_t>(k)))
             .second) {
      return false;
    }
  }
  return true;
}

bool kmer_ok(const std::string &seq, const std::vector<uint8_t> &qual, int start,
             int k, int min_base_quality) {
  for (int i = 0; i < k; ++i) {
    const char base = seq[static_cast<size_t>(start + i)];
    if (base != 'A' && base != 'C' && base != 'G' && base != 'T') {
      return false;
    }
    if (qual[static_cast<size_t>(start + i)] < min_base_quality) {
      return false;
    }
  }
  return true;
}

std::string read_sequence(const bam1_t *read) {
  std::string seq(static_cast<size_t>(read->core.l_qseq), 'N');
  const uint8_t *raw = bam_get_seq(read);
  for (int i = 0; i < read->core.l_qseq; ++i) {
    seq[static_cast<size_t>(i)] = seq_nt16_str[bam_seqi(raw, i)];
  }
  return seq;
}

std::vector<uint8_t> read_qual(const bam1_t *read) {
  const uint8_t *raw = bam_get_qual(read);
  return std::vector<uint8_t>(raw, raw + read->core.l_qseq);
}

Graph build_graph(const std::string &ref, const std::vector<const bam1_t *> &reads,
                  int k, const RealignerOptions &options) {
  Graph graph;
  const int last = static_cast<int>(ref.size()) - k;
  for (int i = 0; i <= last; ++i) {
    const int from = graph.node(ref.substr(static_cast<size_t>(i), static_cast<size_t>(k)));
    if (i < last) {
      const int to =
          graph.node(ref.substr(static_cast<size_t>(i + 1), static_cast<size_t>(k)));
      graph.add_edge(from, to, true);
    }
  }
  graph.source = graph.index[ref.substr(0, static_cast<size_t>(k))];
  graph.sink = graph.index[ref.substr(ref.size() - static_cast<size_t>(k),
                                      static_cast<size_t>(k))];

  for (const bam1_t *read : reads) {
    if (read->core.qual < options.min_mapq) {
      continue;
    }
    const std::string seq = read_sequence(read);
    const std::vector<uint8_t> qual = read_qual(read);
    const int stop = static_cast<int>(seq.size()) - k;
    for (int i = 0; i < stop; ++i) {
      if (!kmer_ok(seq, qual, i, k + 1, options.min_base_quality)) {
        continue;
      }
      const int from = graph.node(seq.substr(static_cast<size_t>(i), static_cast<size_t>(k)));
      const int to =
          graph.node(seq.substr(static_cast<size_t>(i + 1), static_cast<size_t>(k)));
      graph.add_edge(from, to, false);
    }
  }
  return graph;
}

bool has_cycle(const Graph &graph) {
  enum class Color { kWhite, kGray, kBlack };
  std::vector<Color> color(graph.kmers.size(), Color::kWhite);
  bool cycle = false;
  const auto visit = [&](const auto &self, int node) -> void {
    color[static_cast<size_t>(node)] = Color::kGray;
    for (const Graph::Edge &edge : graph.out[static_cast<size_t>(node)]) {
      const Color next = color[static_cast<size_t>(edge.to)];
      if (next == Color::kGray) {
        cycle = true;
      } else if (next == Color::kWhite) {
        self(self, edge.to);
      }
    }
    color[static_cast<size_t>(node)] = Color::kBlack;
  };
  for (int node = 0; node < static_cast<int>(graph.kmers.size()); ++node) {
    if (color[static_cast<size_t>(node)] == Color::kWhite) {
      visit(visit, node);
    }
  }
  return cycle;
}

std::vector<std::string> haplotypes_from(const Graph &graph,
                                         const RealignerOptions &options) {
  struct Path {
    std::vector<int> nodes;
  };
  std::queue<Path> extendable;
  std::vector<Path> done;
  extendable.push(Path{{graph.source}});
  while (!extendable.empty()) {
    if (static_cast<int>(done.size() + extendable.size()) > options.max_num_paths) {
      return {};
    }
    Path path = std::move(extendable.front());
    extendable.pop();
    const int last = path.nodes.back();
    for (const Graph::Edge &edge : graph.out[static_cast<size_t>(last)]) {
      if (!edge.is_ref && edge.weight < options.min_edge_weight) {
        continue;
      }
      Path next = path;
      next.nodes.push_back(edge.to);
      if (edge.to == graph.sink) {
        done.push_back(std::move(next));
      } else {
        extendable.push(std::move(next));
      }
    }
  }

  std::vector<std::string> haplotypes;
  const int k = static_cast<int>(graph.kmers[static_cast<size_t>(graph.source)].size());
  for (const Path &path : done) {
    std::string haplotype;
    for (int node : path.nodes) {
      haplotype.push_back(graph.kmers[static_cast<size_t>(node)][0]);
    }
    haplotype += graph.kmers[static_cast<size_t>(path.nodes.back())].substr(1);
    if (static_cast<int>(haplotype.size()) >= k) {
      haplotypes.push_back(std::move(haplotype));
    }
  }
  std::sort(haplotypes.begin(), haplotypes.end());
  haplotypes.erase(std::unique(haplotypes.begin(), haplotypes.end()),
                   haplotypes.end());
  return haplotypes;
}

std::vector<CigarOp> explode(const std::vector<CigarOp> &cigar) {
  std::vector<CigarOp> steps;
  for (const CigarOp &op : cigar) {
    for (int i = 0; i < op.length; ++i) {
      steps.push_back(CigarOp{op.op, 1});
    }
  }
  return steps;
}

void push_op(std::vector<CigarOp> &cigar, uint32_t op) {
  if (!cigar.empty() && cigar.back().op == op) {
    ++cigar.back().length;
  } else {
    cigar.push_back(CigarOp{op, 1});
  }
}

enum class Trace { kNone, kMatch, kIns, kDel };

struct AlignMatrices {
  int rows = 0;
  int cols = 0;
  std::vector<int> match;
  std::vector<int> ins;
  std::vector<int> del;
  std::vector<uint8_t> from_match;
  std::vector<uint8_t> from_ins;
  std::vector<uint8_t> from_del;

  int index(int i, int j) const { return i * cols + j; }
};

} // namespace

CigarOp align_op(uint32_t op, int length) { return CigarOp{op, length}; }

std::vector<CigarOp> align_affine(const std::string &query,
                                  const std::string &target,
                                  const RealignerOptions &options) {
  const int n = static_cast<int>(query.size());
  const int m = static_cast<int>(target.size());
  AlignMatrices dp;
  dp.rows = n + 1;
  dp.cols = m + 1;
  const int cells = dp.rows * dp.cols;
  dp.match.assign(static_cast<size_t>(cells), kNeg);
  dp.ins.assign(static_cast<size_t>(cells), kNeg);
  dp.del.assign(static_cast<size_t>(cells), kNeg);
  dp.from_match.assign(static_cast<size_t>(cells), 0);
  dp.from_ins.assign(static_cast<size_t>(cells), 0);
  dp.from_del.assign(static_cast<size_t>(cells), 0);
  dp.match[0] = 0;

  for (int i = 1; i <= n; ++i) {
    dp.ins[dp.index(i, 0)] =
        -(options.gap_open + (i - 1) * options.gap_extend);
    dp.from_ins[static_cast<size_t>(dp.index(i, 0))] = i == 1 ? 0 : 1;
  }
  for (int j = 1; j <= m; ++j) {
    dp.del[dp.index(0, j)] =
        -(options.gap_open + (j - 1) * options.gap_extend);
    dp.from_del[static_cast<size_t>(dp.index(0, j))] = j == 1 ? 0 : 1;
  }

  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) {
      const int here = dp.index(i, j);
      const int up = dp.index(i - 1, j);
      const int left = dp.index(i, j - 1);
      const int diag = dp.index(i - 1, j - 1);
      const int open_ins = dp.match[static_cast<size_t>(up)] - options.gap_open;
      const int ext_ins = dp.ins[static_cast<size_t>(up)] - options.gap_extend;
      if (open_ins >= ext_ins) {
        dp.ins[static_cast<size_t>(here)] = open_ins;
        dp.from_ins[static_cast<size_t>(here)] = 0;
      } else {
        dp.ins[static_cast<size_t>(here)] = ext_ins;
        dp.from_ins[static_cast<size_t>(here)] = 1;
      }
      const int open_del = dp.match[static_cast<size_t>(left)] - options.gap_open;
      const int ext_del = dp.del[static_cast<size_t>(left)] - options.gap_extend;
      if (open_del >= ext_del) {
        dp.del[static_cast<size_t>(here)] = open_del;
        dp.from_del[static_cast<size_t>(here)] = 0;
      } else {
        dp.del[static_cast<size_t>(here)] = ext_del;
        dp.from_del[static_cast<size_t>(here)] = 1;
      }
      const int subst =
          query[static_cast<size_t>(i - 1)] == target[static_cast<size_t>(j - 1)]
              ? options.match
              : -options.mismatch;
      int best = dp.match[static_cast<size_t>(diag)];
      uint8_t from = 0;
      if (dp.ins[static_cast<size_t>(diag)] > best) {
        best = dp.ins[static_cast<size_t>(diag)];
        from = 1;
      }
      if (dp.del[static_cast<size_t>(diag)] > best) {
        best = dp.del[static_cast<size_t>(diag)];
        from = 2;
      }
      dp.match[static_cast<size_t>(here)] = best + subst;
      dp.from_match[static_cast<size_t>(here)] = from;
    }
  }

  const int end = dp.index(n, m);
  int state = 0;
  int best = dp.match[static_cast<size_t>(end)];
  if (dp.ins[static_cast<size_t>(end)] > best) {
    best = dp.ins[static_cast<size_t>(end)];
    state = 1;
  }
  if (dp.del[static_cast<size_t>(end)] > best) {
    state = 2;
  }

  std::vector<CigarOp> reversed;
  int i = n;
  int j = m;
  while (i > 0 || j > 0) {
    if (state == 1 || (i > 0 && j == 0)) {
      push_op(reversed, BAM_CINS);
      const uint8_t from = dp.from_ins[static_cast<size_t>(dp.index(i, j))];
      --i;
      state = from == 0 ? 0 : 1;
    } else if (state == 2 || (j > 0 && i == 0)) {
      push_op(reversed, BAM_CDEL);
      const uint8_t from = dp.from_del[static_cast<size_t>(dp.index(i, j))];
      --j;
      state = from == 0 ? 0 : 2;
    } else {
      push_op(reversed, BAM_CMATCH);
      const uint8_t from = dp.from_match[static_cast<size_t>(dp.index(i, j))];
      --i;
      --j;
      state = from;
    }
  }
  std::reverse(reversed.begin(), reversed.end());
  std::vector<CigarOp> cigar;
  for (const CigarOp &op : reversed) {
    if (!cigar.empty() && cigar.back().op == op.op) {
      cigar.back().length += op.length;
    } else {
      cigar.push_back(op);
    }
  }
  return cigar;
}

std::vector<CigarOp> compose_cigars(const std::vector<CigarOp> &read_to_hap,
                                    const std::vector<CigarOp> &hap_to_ref) {
  const std::vector<CigarOp> read_steps = explode(read_to_hap);
  const std::vector<CigarOp> hap_steps = explode(hap_to_ref);
  std::vector<CigarOp> cigar;
  size_t a = 0;
  size_t b = 0;
  while (a < read_steps.size() || b < hap_steps.size()) {
    if (a < read_steps.size() && read_steps[a].op == BAM_CINS) {
      push_op(cigar, BAM_CINS);
      ++a;
      continue;
    }
    if (b < hap_steps.size() && hap_steps[b].op == BAM_CDEL) {
      push_op(cigar, BAM_CDEL);
      ++b;
      continue;
    }
    if (a >= read_steps.size() || b >= hap_steps.size()) {
      break;
    }
    const uint32_t read_op = read_steps[a].op;
    const uint32_t hap_op = hap_steps[b].op;
    if (read_op == BAM_CMATCH && hap_op == BAM_CMATCH) {
      push_op(cigar, BAM_CMATCH);
    } else if (read_op == BAM_CMATCH && hap_op == BAM_CINS) {
      push_op(cigar, BAM_CINS);
    } else if (read_op == BAM_CDEL && hap_op == BAM_CMATCH) {
      push_op(cigar, BAM_CDEL);
    }
    ++a;
    ++b;
  }
  return cigar;
}

std::vector<std::pair<int64_t, int64_t>>
assembly_windows(const std::vector<int64_t> &positions, int min_windows_distance,
                 int64_t ref_len) {
  std::vector<int64_t> ordered = positions;
  std::sort(ordered.begin(), ordered.end());
  ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
  std::vector<std::pair<int64_t, int64_t>> windows;
  int64_t cluster_start = -1;
  int64_t cluster_end = -1;
  const auto flush = [&]() {
    if (cluster_start < 0) {
      return;
    }
    const int64_t start =
        std::max<int64_t>(0, cluster_start - min_windows_distance);
    const int64_t end =
        std::min(ref_len, cluster_end + min_windows_distance);
    if (end > start) {
      windows.emplace_back(start, end);
    }
  };
  for (int64_t pos : ordered) {
    if (cluster_start < 0) {
      cluster_start = pos;
      cluster_end = pos;
    } else if (pos > cluster_end + 2 * static_cast<int64_t>(min_windows_distance)) {
      flush();
      cluster_start = pos;
      cluster_end = pos;
    } else {
      cluster_end = pos;
    }
  }
  flush();
  return windows;
}

std::vector<std::string>
assemble_haplotypes(const std::string &ref,
                    const std::vector<const bam1_t *> &reads,
                    const RealignerOptions &options) {
  if (ref.size() < 2) {
    return {};
  }
  const int max_k =
      std::min(options.max_k, static_cast<int>(ref.size()) - 1);
  int min_k = -1;
  for (int k = options.min_k; k <= max_k; k += std::max(options.step_k, 1)) {
    if (ref_kmer_unique(ref, k)) {
      min_k = k;
      break;
    }
  }
  if (min_k < 0) {
    return {};
  }
  for (int k = min_k; k <= max_k; k += std::max(options.step_k, 1)) {
    const Graph graph = build_graph(ref, reads, k, options);
    if (graph.source < 0 || has_cycle(graph)) {
      continue;
    }
    return haplotypes_from(graph, options);
  }
  return {};
}

namespace {

int64_t read_end(const bam1_t *read) {
  return read->core.pos +
         bam_cigar2rlen(read->core.n_cigar, bam_get_cigar(read));
}

std::vector<uint32_t> original_cigar(const bam1_t *read) {
  const uint32_t *cigar = bam_get_cigar(read);
  return std::vector<uint32_t>(cigar, cigar + read->core.n_cigar);
}

std::vector<uint32_t> pack_cigar(const std::vector<CigarOp> &cigar) {
  std::vector<uint32_t> packed;
  packed.reserve(cigar.size());
  for (const CigarOp &op : cigar) {
    packed.push_back(bam_cigar_gen(static_cast<uint32_t>(op.length), op.op));
  }
  return packed;
}

int variant_support(const LocusCount &locus) {
  int support = locus.soft_clip;
  support += locus.alt[0] + locus.alt[1] + locus.alt[2] + locus.alt[3];
  for (const IndelAllele &indel : locus.indels) {
    support += indel.count;
  }
  return support;
}

} // namespace

std::vector<AlignedRead>
realign_reads(const std::string &ref, int64_t ref_start,
              const std::vector<const bam1_t *> &reads,
              const RealignerOptions &options) {
  std::vector<AlignedRead> aligned(reads.size());
  for (size_t i = 0; i < reads.size(); ++i) {
    aligned[i].pos = reads[i]->core.pos;
    aligned[i].cigar = original_cigar(reads[i]);
  }
  if (ref.empty() || reads.empty()) {
    return aligned;
  }

  std::vector<LocusCount> counts;
  for (const bam1_t *read : reads) {
    accumulate_snps(read, ref_start, ref.data(), static_cast<int64_t>(ref.size()),
                    counts);
  }
  std::vector<int64_t> positions;
  for (size_t i = 0; i < counts.size(); ++i) {
    const int support = variant_support(counts[i]);
    if (support >= options.min_supporting_reads &&
        support <= options.max_supporting_reads) {
      positions.push_back(static_cast<int64_t>(i));
    }
  }
  const std::vector<std::pair<int64_t, int64_t>> windows = assembly_windows(
      positions, options.min_windows_distance, static_cast<int64_t>(ref.size()));

  std::vector<char> done(reads.size(), 0);
  for (const auto &window : windows) {
    if (window.second - window.first > options.max_window_size) {
      continue;
    }
    const int64_t abs_start = ref_start + window.first;
    const int64_t abs_end = ref_start + window.second;
    const std::string window_ref = ref.substr(
        static_cast<size_t>(window.first),
        static_cast<size_t>(window.second - window.first));
    std::vector<const bam1_t *> window_reads;
    std::vector<size_t> window_index;
    for (size_t i = 0; i < reads.size(); ++i) {
      if (done[i] || reads[i]->core.l_qseq > options.max_read_length) {
        continue;
      }
      const int64_t end = read_end(reads[i]);
      if (reads[i]->core.pos < abs_end && end > abs_start) {
        window_reads.push_back(reads[i]);
        window_index.push_back(i);
      }
    }
    const std::vector<std::string> haplotypes =
        assemble_haplotypes(window_ref, window_reads, options);
    if (haplotypes.size() < 2) {
      continue;
    }
    for (size_t n = 0; n < window_reads.size(); ++n) {
      const std::string sequence = read_sequence(window_reads[n]);
      int best_score = kNeg;
      std::string best_hap;
      for (const std::string &hap : haplotypes) {
        const std::vector<CigarOp> to_hap =
            align_affine(sequence, hap, options);
        int score = 0;
        size_t q = 0;
        size_t t = 0;
        for (const CigarOp &op : to_hap) {
          for (int k = 0; k < op.length; ++k) {
            if (op.op == BAM_CMATCH) {
              score += sequence[q] == hap[t] ? options.match : -options.mismatch;
              ++q;
              ++t;
            } else if (op.op == BAM_CINS) {
              score -= k == 0 ? options.gap_open : options.gap_extend;
              ++q;
            } else {
              score -= k == 0 ? options.gap_open : options.gap_extend;
              ++t;
            }
          }
        }
        if (score > best_score) {
          best_score = score;
          best_hap = hap;
        }
      }
      if (best_hap.empty()) {
        continue;
      }
      const std::vector<CigarOp> read_to_hap =
          align_affine(sequence, best_hap, options);
      const std::vector<CigarOp> hap_to_ref =
          align_affine(best_hap, window_ref, options);
      const std::vector<CigarOp> cigar = compose_cigars(read_to_hap, hap_to_ref);
      if (cigar.empty()) {
        continue;
      }
      aligned[window_index[n]].pos = abs_start;
      aligned[window_index[n]].cigar = pack_cigar(cigar);
      aligned[window_index[n]].realigned = true;
      done[window_index[n]] = 1;
    }
  }
  return aligned;
}

} // namespace make_examples
} // namespace rodv
