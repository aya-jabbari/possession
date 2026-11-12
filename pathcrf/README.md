# Pretrained PathCRF experiment

This adapter runs the official pretrained dynamic masked CRF (`saved/140`) on
the supplied tactical ATD. It does not train a classifier or open answer keys
during prediction. It is an experiment on noisy tracking, not a guarantee of
high-quality possession labels.

The full-match run uses the user-approved **positional goalkeeper fallback**.
YOLO is evaluated independently on short clips, on every frame in those clips,
without a tracker. The full-match predictions do not use those YOLO detections.

## Run from the possession directory

```powershell
python -m pip install -r requirements-pathcrf.txt
python -B -X utf8 -m unittest pathcrf_pipeline.test_pipeline -v
python -B -u -X utf8 -m pathcrf_pipeline.run --threads 4
python -B -u -X utf8 -m pathcrf_pipeline.goalkeepers
python -B -u -X utf8 -m pathcrf_pipeline.compare --bootstrap 1000
```

On the development machine PyTorch Geometric was installed locally under
`pathcrf/dependencies/`; the adapter also accepts a normal environment install.
The original GBDT pipeline, its artifacts and the published baseline CSVs are
not changed by these commands. Output is isolated under `pathcrf/output/`.

Restore missing source/checkpoints with:

```powershell
python -B -m pathcrf_pipeline.fetch_assets --with-detector
```

The downloader pins the source revision and verifies Git blob hashes. It checks
the detector weights against a fixed SHA-256 and refuses to overwrite changed
files. It requires internet access. Derived competition data and local dependency
files are ignored by git.

## Data preparation

- Every raw ATD row must exist exactly once. `frame_id - 1` is the zero-based
  tactical video/submission index. The cache is invalidated by input SHA-256.
- Coordinates are mapped from the supplied 0-1 pitch system to the model's
  105 x 68 metre pitch. Pixel normalization is not pitch normalization.
- Internal position gaps of at most five frames are interpolated; leading,
  trailing and longer gaps remain missing. Each inference window uses only
  slots with complete resulting trajectories. A >2 metre per-frame jump or
  >20 m/s smoothed speed excludes the slot. These checks cannot detect every
  identity switch, especially switches between nearby players.
- Near-identical tracks are removed only when their positions remain within
  0.75 m for more than 80% of the window. An unresolved excess of more than 11
  slots per team rejects the window instead of truncating arbitrary tracks.
- The minimum is eight continuous players per team. Missing players use the
  upstream zero-padding convention; decoder states involving padding are masked.
  This incomplete-player input is a domain shift and is counted in the audit.
- Team blocks follow left/right keeper positions. The positional goalkeeper
  heuristic is the upstream farthest-average-x-from-midfield rule, per window.
- Velocity is a 25 FPS position difference followed by the upstream 15-point,
  order-2 Savitzky-Golay smoothing. Acceleration components use another
  difference and 9-point, order-2 smoothing. `accel` is their vector magnitude.
  `speed` is recomputed from velocity, keeping all motion features consistent.
- Node features are `[team_indicator, goalkeeper_indicator, x, y, vx, vy,
  speed, accel]`. Four fixed outside nodes are appended. Input is float32.
- Fixed heuristics estimate dead-ball intervals from observed outside balls or
  sustained stationary/unattended balls. Ball absence alone never marks D. An
  observed outside interval may be extended to a supported restart within 30 s.
  Smart Tags support control evidence; their broad intervals are not copied as
  possession labels. The dead-ball rules are incomplete, including for fouls.
- Estimated-live segments are windowed at 10 seconds with five-second overlap,
  without crossing half boundaries or estimated-dead intervals. Original 25 FPS
  features are sampled once to 5 FPS (the final endpoint is retained).

## Inference and outputs

Weights load strictly from `saved/140/model/state_dict_best_acc.pt`. The adapter
uses the unchanged official network and dynamic sparse CRF decoder, with dropout
disabled. It checks that decoded paths obey the model's hard transition rules.
The input/decoder test matches the official `inference_episode(evaluate=False)`
on the same complete synthetic tracking input.

Decoded sender team retains possession during flight. A player becoming the
source of the next state changes ownership. Outside self-loops map to D. The
result is an approximation of the challenge's controlled-possession definition;
the model does not distinguish every deflection from genuine control.
Categorical predictions are forward-held back onto 25 FPS and blended by
triangular votes in overlapping windows. CRF paths reset between windows; there
is no claim of a single globally consistent full-match player-level path.
This is offline inference: attention and Viterbi use future context within each
window. It is not a causal live-broadcast system.

`output/results/` contains three complete 199,319-row files:

| File | Meaning |
| --- | --- |
| `pathcrf_model.csv` | Model/CRF output on accepted estimated-live windows; X elsewhere |
| `rules_only.csv` | Automatic geometry/tag control evidence with last-team retention and estimated D |
| `pathcrf_rules.csv` | Model output where covered, rules on gaps, and heuristic D override |

The `pathcrf_model` entry shares estimated-live segmentation with the rule layer;
it is not an experiment where the network independently detects all stoppages.
An X counts as a miss for macroF1 and is filled by the official spell scorer.

Audits include `run_summary.json`, `window_audit.json`, `frame_audit.parquet`,
`decoded_edges.parquet` and raw vote arrays. The prediction process installs an
audit hook denying all opens inside `evaluation/gt/`. It never loads old GBDT
artifacts or uses the public passage to fit/tune parameters.

The separate comparison command uses only the public five-minute answer key. It
scores the four supplied baseline CSVs and the three new outputs with the
unchanged official evaluator, writes bootstrap intervals and prediction hashes,
and creates `output/comparison/COMPARISON.md`, `scoreboard.csv` and
`public_timeline.png`. Public-window scores cannot establish full-match accuracy.
This passage was already inspected during historical development of the project,
so it is not an untouched test set.

## Independent YOLO check

Default half-open clips are 12000:12025, 45000:45025, 113742:113767 and
150000:150025: 100 frames across both halves. Override them with `--clips`.
YOLO prediction mode runs at 640 px and confidence 0.25. There is no `.track()`
call or persistence of identities between detections. Detection bottom-centres
are inverse-projected using the same frame's homography and matched to ATD slots
within 4 m only if the next-nearest slot is at least 1 m farther away. Unmatched
or ambiguous detections remain unmatched. These are matching heuristics, not a
measured identification accuracy. Visual samples and raw detections stay local
under the ignored output folder.

## Sources

- PathCRF: https://github.com/hyunsungkim-ds/pathcrf, revision
  `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`; original MPL-2.0 license in
  `upstream/LICENSE`. Vendored source is unmodified.
- Goalkeeper detector: https://huggingface.co/martinjolif/yolo-football-player-detection,
  revision `5e83fafa8d564243001ce8e063612a618a138fbe`. Its AGPL-3.0 model card
  and provenance are preserved in `detector/`.

No match assets, tracking exports or clipped images should be redistributed.
