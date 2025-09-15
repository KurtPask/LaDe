#!/bin/bash
#SBATCH --job-name=tropical_cproute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ROUTE_PRED_DIR="$PROJECT_ROOT/route_prediction"
LOG_DIR=${LOG_DIR:-"$ROUTE_PRED_DIR/logs"}
VENV_DIR=${VENV_DIR:-"$PROJECT_ROOT/.venvs/tropical_cproute"}
PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET_NAME=${DATASET_NAME:-pickup_yt_0614_dataset_change}
HF_REPO_ID=${HF_REPO_ID:-Cainiao-AI/LaDe-P}
HF_REPO_PREFIX=${HF_REPO_PREFIX:-route_prediction/dataset}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCH=${NUM_EPOCH:-30}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
SORT_X_SIZE=${SORT_X_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
EARLY_STOP=${EARLY_STOP:-5}

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting tropical CPRoute job"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "ROUTE_PRED_DIR=$ROUTE_PRED_DIR"
echo "VENV_DIR=$VENV_DIR"

cd "$PROJECT_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[$(date)] Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROUTE_PRED_DIR/requirements.txt"
python -m pip install "huggingface_hub==0.34.5"

DATASET_DIR="$ROUTE_PRED_DIR/data/dataset/$DATASET_NAME"
if [[ "${SKIP_DATA_DOWNLOAD:-0}" != "1" ]]; then
    echo "[$(date)] Downloading dataset splits into $DATASET_DIR"
    python "$ROUTE_PRED_DIR/scripts/download_pickup_dataset.py" \
        --dataset "$DATASET_NAME" \
        --output-dir "$DATASET_DIR" \
        --repo-id "$HF_REPO_ID" \
        --repo-prefix "$HF_REPO_PREFIX"
else
    echo "[$(date)] Skipping dataset download as requested."
fi

extra_args=()
if [[ -n "${TROPICAL_KWARGS_JSON:-}" ]]; then
    extra_args+=(--tropical_attention_kwargs "$TROPICAL_KWARGS_JSON")
fi
if [[ -n "${ADDITIONAL_RUN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    additional_args=( ${ADDITIONAL_RUN_ARGS} )
    extra_args+=("${additional_args[@]}")
fi

echo "[$(date)] Launching CPRoute training with tropical attention"
cmd=(
    python "$ROUTE_PRED_DIR/run.py"
    --model cproute
    --dataset "$DATASET_NAME"
    --batch_size "$BATCH_SIZE"
    --num_epoch "$NUM_EPOCH"
    --hidden_size "$HIDDEN_SIZE"
    --sort_x_size "$SORT_X_SIZE"
    --lr "$LEARNING_RATE"
    --wd "$WEIGHT_DECAY"
    --early_stop "$EARLY_STOP"
    --use_tropical_attention
)
if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
fi
"${cmd[@]}"

echo "[$(date)] Tropical CPRoute job completed"
