#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results_final}"
TAG="${TAG:-internal_tau}"
TRAIN_END="${TRAIN_END:-2019-12-31}"
VAL_END="${VAL_END:-2020-12-31}"
TAU_TAGS="${TAU_TAGS:-}"

if [[ -z "${TAU_TAGS}" ]]; then
  mapfile -t found_tau_dirs < <(find data/processed -maxdepth 1 -type d -name 'tau_*' | sort)
  if [[ ${#found_tau_dirs[@]} -eq 0 ]]; then
    echo "ERROR: aucun dossier data/processed/tau_* trouvé" >&2
    exit 1
  fi

  tau_names=()
  for d in "${found_tau_dirs[@]}"; do
    tau_names+=("$(basename "$d")")
  done

  IFS=,
  TAU_TAGS="${tau_names[*]}"
  unset IFS
fi

IFS=',' read -r -a tau_list <<< "${TAU_TAGS}"
first_tau="${tau_list[0]}"
labels_path="data/processed/${first_tau}/labels.csv"
raw_prices_path="data/raw/adj_close_2014-01-01_2024-12-31.csv"

if [[ ! -f "${labels_path}" ]]; then
  echo "ERROR: labels_path introuvable: ${labels_path}" >&2
  exit 1
fi
if [[ ! -f "${raw_prices_path}" ]]; then
  echo "ERROR: raw_prices_path introuvable: ${raw_prices_path}" >&2
  exit 1
fi

rm -rf "${RESULTS_DIR}"

echo "=== tag=${TAG} tau_tags=${TAU_TAGS} ==="

echo "=== graph models (tau interne) ==="
python -m src.pipeline.train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel wl --model krr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --tau_tags "${TAU_TAGS}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel wl --model svr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --tau_tags "${TAU_TAGS}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel sp --model krr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --tau_tags "${TAU_TAGS}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel sp --model svr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --tau_tags "${TAU_TAGS}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"

echo "=== baselines ==="
python -m src.pipeline.baseline_train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --model ridge --labels_path "${labels_path}" --raw_prices_path "${raw_prices_path}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.baseline_train_eval --results_dir "${RESULTS_DIR}" --tag "${TAG}" --model mean --labels_path "${labels_path}" --raw_prices_path "${raw_prices_path}" --train_end "${TRAIN_END}" --val_end "${VAL_END}"

echo "=== summarize ==="
python -m src.pipeline.summarize_results --results_dir "${RESULTS_DIR}"

echo "=== benchmark inference ==="
python -m src.pipeline.benchmark_inference --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel wl --model krr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.benchmark_inference --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel wl --model svr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.benchmark_inference --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel sp --model krr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --train_end "${TRAIN_END}" --val_end "${VAL_END}"
python -m src.pipeline.benchmark_inference --results_dir "${RESULTS_DIR}" --tag "${TAG}" --kernel sp --model svr --labels_path "${labels_path}" --graphs_root data/graphs --processed_root data/processed --train_end "${TRAIN_END}" --val_end "${VAL_END}"

echo "DONE"
