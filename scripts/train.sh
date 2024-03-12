OUTPUT_DIR=$1
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

python train_net_video.py --config configs/refcoco/swin/swin_base.yaml --num-gpus 2 OUTPUT_DIR ${OUTPUT_DIR} ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"
