#!/bin/bash
# Fetch a SONIC ONNX export from the training machine and score it on the 26 crawl clips.
# usage: TRAIN_HOST=user@host ROOT=<workspace with HumanTracker/ and data/> score_sonic.sh <remote dir holding exported/*_encoder.onnx and *_decoder.onnx> <name>
set -euo pipefail
ROOT=${ROOT:?set ROOT to the workspace holding HumanTracker/ and data/}
TRAIN_HOST=${TRAIN_HOST:?set TRAIN_HOST to the training machine, as user@host}
DST=$ROOT/runs/exports/$2
mkdir -p "$DST" "$ROOT/runs/scores"
rsync -az --include '*_encoder.onnx' --include '*_decoder.onnx' --exclude '*' "$TRAIN_HOST:$1/exported/" "$DST/"
# The benchmark's backend expects the released 1762-D encoder input.
uv run -q --no-project --python 3.12 --with onnx python "$ROOT/tools/sonic_pad_encoder.py" "$(ls $DST/*_encoder.onnx | grep -v padded | tail -1)" "$DST/padded_encoder.onnx" > /dev/null
cd "$ROOT/HumanTracker"
.venv/bin/python -m humantracker.eval.eval_parallel_tracker --tracker sonic \
    --encoder "$DST/padded_encoder.onnx" --decoder "$(ls $DST/*_decoder.onnx | tail -1)" \
    --device cpu --rm_device cpu --mocap_path ../data/HumanTracker/motions \
    --test_json ../data/HumanTracker/motions/test_crawl.json --termination_metric whole_body \
    --workers 10 --video_interval 0 --rollout_dir "$ROOT/runs/rollouts/$2" --rollout_run_id "$2" \
    --output_json "$ROOT/runs/scores/$2.json" > "$ROOT/runs/scores/$2.log" 2>&1
python3 -I -c "
import json
d = json.load(open('$ROOT/runs/scores/$2.json'))
t = d['per_trajectory']; t = list(t.values()) if isinstance(t, dict) else t
s = d['category_summaries'][0]
print('$2: crawl success %d/%d, mean progress %.2f, MPJPE %.3f' % (sum(x['length_ratio'] >= 0.999 for x in t), len(t), sum(x['length_ratio'] for x in t) / len(t), s['MPJPE']))"
