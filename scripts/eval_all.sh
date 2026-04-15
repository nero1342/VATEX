#!/usr/bin/env bash
# Evaluate all 3 models across their test splits on GPU 2.
# All 8 jobs run in parallel — GPU 2 has 97 GB, each job uses ~2-3 GB.

CUDA_DEV=2
CONFIG="configs/refcoco/swin/swin_base.yaml"

run_eval() {
    local weight=$1
    local dataset=$2
    local outdir=$3

    mkdir -p "${outdir}"
    echo "[$(date '+%H:%M:%S')] START  ${dataset}  (${weight})"

    CUDA_VISIBLE_DEVICES=${CUDA_DEV} python train_net_video.py \
        --config ${CONFIG} \
        --num-gpus 1 \
        --eval-only \
        MODEL.WEIGHTS "${weight}" \
        DATASETS.TEST "(\"${dataset}\",)" \
        OUTPUT_DIR "${outdir}" \
        > "${outdir}/eval.log" 2>&1

    echo "[$(date '+%H:%M:%S')] DONE   ${dataset}  →  ${outdir}/eval.log"
}

# Launch all 8 splits in parallel
run_eval weights/refcoco_final.pth   refcoco_val      output/eval/refcoco/val   &
run_eval weights/refcoco_final.pth   refcoco_testA    output/eval/refcoco/testA &
run_eval weights/refcoco_final.pth   refcoco_testB    output/eval/refcoco/testB &

run_eval "weights/refcoco+_139999.pth" "refcoco+_val"   output/eval/refcoco+/val   &
run_eval "weights/refcoco+_139999.pth" "refcoco+_testA" output/eval/refcoco+/testA &
run_eval "weights/refcoco+_139999.pth" "refcoco+_testB" output/eval/refcoco+/testB &

run_eval weights/refcocog_139999.pth refcocog_val     output/eval/refcocog/val  &
run_eval weights/refcocog_139999.pth refcocog_test    output/eval/refcocog/test &

wait
echo ""
echo "=========================================="
echo " All evaluations complete — results below"
echo "=========================================="

for log in \
    output/eval/refcoco/val/eval.log \
    output/eval/refcoco/testA/eval.log \
    output/eval/refcoco/testB/eval.log \
    output/eval/refcoco+/val/eval.log \
    output/eval/refcoco+/testA/eval.log \
    output/eval/refcoco+/testB/eval.log \
    output/eval/refcocog/val/eval.log \
    output/eval/refcocog/test/eval.log; do
    echo ""
    echo "--- ${log} ---"
    grep -E "oIoU|mIoU|cIoU|Overall|Precision|overall_iou|mean_iou" "${log}" 2>/dev/null | tail -10 \
        || echo "(no results yet)"
done
