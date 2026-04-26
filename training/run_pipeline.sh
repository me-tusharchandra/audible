#!/usr/bin/env bash
#
# Post-baseline-training pipeline — runs hands-off:
#   1. Eval the most recent baseline against the env (-> baseline metrics)
#   2. Run the adaptive-curriculum loop (3 rounds: eval → mine → generate → retrain)
#   3. Render plots (dataset distribution, reward curve over rounds, confusion matrix)
#
# All output streams to stdout AND to the appropriate JSON/PNG files. Each
# step is idempotent and writes to disk before the next, so a crash after
# the first eval still leaves baseline metrics on disk.

set -e

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHONPATH="${REPO_ROOT}"
export PYTHONPATH

echo "=================================================================="
echo "AUDIBLE — post-baseline pipeline"
echo "repo: ${REPO_ROOT}"
echo "=================================================================="

LATEST_RUN=$(ls -dt training/runs/*/ 2>/dev/null | head -1 | sed 's:/$::')
if [ -z "${LATEST_RUN}" ]; then
    echo "ERROR: no training runs found in training/runs/"
    exit 1
fi
echo "baseline model: ${LATEST_RUN}"

echo
echo "==[ STEP 1: baseline env-rollout eval ]==========================="
"${PYTHON}" -m training.eval_env --model "${LATEST_RUN}" --rollouts 300 --port 8765 2>&1 | tail -50
echo
echo "baseline metrics written to ${LATEST_RUN}/env_eval.json"

echo
echo "==[ STEP 2: adaptive-curriculum loop (3 rounds) ]================="
"${PYTHON}" -m training.curriculum \
    --rounds 3 \
    --gen-per-round 100 \
    --rollouts-per-eval 200 \
    --epochs-per-round 1 \
    --port 8766 2>&1 | tee "${REPO_ROOT}/training/curriculum.log" | tail -100

echo
echo "==[ STEP 3: rendering plots ]====================================="
"${PYTHON}" -c "
from pathlib import Path
import json
from training.plots import plot_dataset_distribution, plot_reward_curve, plot_confusion
from training.dataset import CLASS_NAMES

print('1/3 dataset_distribution...')
plot_dataset_distribution()

curriculum_log = Path('audible_env/data/curriculum/curriculum_log.json')
if curriculum_log.exists():
    print('2/3 reward_curve...')
    plot_reward_curve(curriculum_log)
else:
    print('2/3 SKIP — curriculum log missing')

# Final-checkpoint confusion matrix from the most recent run
runs = sorted(Path('training/runs').glob('*'))
metrics = runs[-1] / 'eval_metrics.json'
if metrics.exists():
    print(f'3/3 confusion_matrix from {runs[-1].name}...')
    plot_confusion(metrics, CLASS_NAMES)
else:
    print('3/3 SKIP — eval_metrics.json missing')
print('plots in training/plots/')
"

echo
echo "==[ DONE ]========================================================"
echo "Artifacts:"
echo "  - baseline env eval:    ${LATEST_RUN}/env_eval.json"
echo "  - curriculum log:       audible_env/data/curriculum/curriculum_log.json"
echo "  - plots:                training/plots/*.png"
echo
echo "Next step:  openenv push   (deploy env to HF Space)"
