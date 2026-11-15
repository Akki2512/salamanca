# Crawl specialist for the Unitree G1

SONIC fine-tuned on crawl data that includes our own Rokoko suit recordings. Scored on the
26 crawl clips of the HumanTracker benchmark (ECCV 2026) with the benchmark's evaluator.

## Result

| Policy | Crawl clips completed (of 26) | Mean progress | Joint error (rad) |
| --- | --- | --- | --- |
| Humanoid-GPT | 0 | 15% | not measured on crawl |
| SONIC 1.1 | 3 | 40% | 0.254 |
| SONIC (released) | 9 | 55% | 0.260 |
| **This model** | **18** | **80%** | **0.208** |

Against SONIC on the same clips: 10 gained, 1 lost, exact McNemar p = 0.012.
Per-clip numbers are in `results.json`.

## Limits

- **Crawl only.** On all 164 Ground clips it completes 27, against 32 for released SONIC.
  It gained 9 crawls and lost 15 stand-sit clips.
- **Five leaderboard policies are unscored on crawl:** HoloMotion, ScaleBFM-M, ScaleBFM-XL,
  HEFT, TeleopIT. The other nine cannot exceed 18.
- **The suit data's share of the gain is unmeasured.** One run, one seed.

## How it was made

1. Three Rokoko takes of crawling, 4.3 minutes (`data/suit_recordings`).
2. Retargeted to the G1 with GMR, with a floor-height fix (`tools/retarget_rokoko.py`,
   `tools/process_clips.py`).
3. Cut to 12 s hands-and-knees windows, mirrored, with 1.5x and 2x speed copies.
4. Nemotron 3.5 Super VL labelled rendered frames of each clip; only clips it called
   `crawl` were kept: 90 of 132 suit clips and 76 of 86 public clips (`tools/nemotron_label.py`).
5. SONIC's released checkpoint fine-tuned on those 166 clips for 1,500 updates on one B300
   (`tools/finetune.sh`).
6. Exported to ONNX and scored (`tools/export.sh`, `tools/sonic_pad_encoder.py`,
   `tools/score_sonic.sh`).

## Contents

| Path | What |
| --- | --- |
| `model/encoder.onnx`, `model/decoder.onnx` | The policy, in the layout the benchmark scores |
| `data/suit_recordings` | The three raw Rokoko BVH takes |
| `data/train_clips/suit`, `data/train_clips/public` | The 166 training clips, G1 qpos at 50 Hz |
| `videos/vs_humanoid_gpt.mp4` | Reference, Humanoid-GPT, this model |
| `videos/sonic_before_vs_after.mp4` | Reference, released SONIC, this model |
| `results.json` | Scores per policy and per clip |
| `tools` | The scripts for each step above |

The git branch holds `README.md`, `results.json` and `tools` only. `model`, `data` and `videos`
are kept out of git under the team rule against weights, datasets, videos and files over 5 MB.

## Reproducing the score

With the HumanTracker repo and its test motions:

```bash
python -m humantracker.eval.eval_parallel_tracker --tracker sonic \
    --encoder model/encoder.onnx --decoder model/decoder.onnx \
    --device cpu --rm_device cpu --termination_metric whole_body \
    --mocap_path <HumanTracker>/motions --test_json <crawl-only test.json>
```
