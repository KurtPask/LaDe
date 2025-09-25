#!/bin/bash
#SBATCH --job-name=tropical_cproute
#SBATCH --partition=dsag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

. /etc/profile
module load lang/python/3.13.0

source "$HOME/LaDe/LaDe/.venvs/tropical_cproute/bin/activate"

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DATA_SCRIPT="$PROJECT_ROOT/data/dataset.py"
RAW_ROOT="$PROJECT_ROOT/data/raw"
TMP_ROOT="$PROJECT_ROOT/data/tmp"

run_dataset() {
  local split="$1"
  local csv_path="$2"
  local base_name
  base_name="$(basename "${csv_path%.csv}")"
  local tmp_dir="$TMP_ROOT/$base_name"

  echo "[$(date --iso-8601=seconds)] Processing ${split} dataset: $base_name"
  python "$DATA_SCRIPT" \
    --fin_original "$csv_path" \
    --fin_temp "$tmp_dir" \
    --data_name "$base_name"
}

process_split_dir() {
  local split="$1"
  local dir_path="$2"

  if [ ! -d "$dir_path" ]; then
    echo "Directory $dir_path does not exist; skipping $split datasets."
    return
  fi

  local csv_files=("$dir_path"/*.csv)

  if [ ${#csv_files[@]} -eq 0 ]; then
    echo "No CSV files found in $dir_path; skipping $split datasets."
    return
  fi

  for csv_file in "${csv_files[@]}"; do
    run_dataset "$split" "$csv_file"
  done
}

process_split_dir "pickup" "$RAW_ROOT/pickup"
process_split_dir "delivery" "$RAW_ROOT/delivery"

echo "[$(date --iso-8601=seconds)] Dataset preprocessing complete."
