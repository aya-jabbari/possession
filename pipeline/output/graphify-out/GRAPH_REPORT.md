# Graph Report - pipeline  (2026-11-09)

## Corpus Check
- 8 files · ~178,477 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: .npy 1, .parquet 1)

## Summary
- 73 nodes · 152 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c38134fc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- viterbi.py
- run_pipeline
- numpy
- gbdt_model.py
- check_gbdt.py

## God Nodes (most connected - your core abstractions)
1. `run_pipeline()` - 17 edges
2. `check_holdout()` - 12 edges
3. `build_features()` - 9 edges
4. `check_saved()` - 8 edges
5. `train_gbdt()` - 8 edges
6. `load_model()` - 7 edges
7. `extract_features()` - 6 edges
8. `save_model()` - 6 edges
9. `load_team_assignments()` - 5 edges
10. `parse_smart_tagging()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Testing the possession model` --references--> `train_gbdt()`  [INFERRED]
  pipeline/README.md → pipeline/gbdt_model.py
- `check_saved()` --calls--> `load_ground_truth()`  [INFERRED]
  pipeline/check_gbdt.py → pipeline/gbdt_model.py
- `check_saved()` --calls--> `load_model()`  [INFERRED]
  pipeline/check_gbdt.py → pipeline/gbdt_model.py
- `check_saved()` --calls--> `predict_emissions()`  [INFERRED]
  pipeline/check_gbdt.py → pipeline/gbdt_model.py
- `check_holdout()` --calls--> `load_model()`  [INFERRED]
  pipeline/check_gbdt.py → pipeline/gbdt_model.py

## Import Cycles
- None detected.

## Communities (6 total, 1 thin omitted)

### Community 0 - "viterbi.py"
Cohesion: 0.13
Nodes (6): build_transition_matrix(), decode_match(), labels_to_submission(), remove_short_segments(), viterbi(), viterbi_vectorised()

### Community 1 - "run_pipeline"
Cohesion: 0.20
Nodes (3): load_ground_truth(), run_pipeline(), write_submission()

### Community 2 - "numpy"
Cohesion: 0.23
Nodes (6): build_features(), _empty_smart_tagging(), extract_features(), load_team_assignments(), parse_atd_raw(), parse_smart_tagging()

### Community 3 - "gbdt_model.py"
Cohesion: 0.17
Nodes (5): load_model(), predict_emissions(), save_model(), train_gbdt(), Testing the possession model

### Community 4 - "check_gbdt.py"
Cohesion: 0.29
Nodes (4): check_holdout(), check_probabilities(), check_saved(), main()

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `run_pipeline` to `viterbi.py`, `numpy`, `gbdt_model.py`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `run_pipeline()` (e.g. with `build_features()` and `load_ground_truth()`) actually correct?**
  _`run_pipeline()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Should `viterbi.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Why does `train_gbdt()` connect `gbdt_model.py` to `run_pipeline`, `numpy`, `check_gbdt.py`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `check_holdout()` (e.g. with `load_model()` and `predict_emissions()`) actually correct?**
  _`check_holdout()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Why does `check_holdout()` connect `check_gbdt.py` to `viterbi.py`, `numpy`, `gbdt_model.py`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._