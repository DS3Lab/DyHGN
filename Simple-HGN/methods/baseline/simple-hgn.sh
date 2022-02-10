export EPOCH=${EPOCH:-1024}
export NUM_LAYERS=${NUM_LAYERS:-2}
export NUM_HEADS=${NUM_HEADS:-4}
export N_HID=${N_HID:-256}
export DROPOUT=${DROPOUT:-0.1}
export FEATS_TYPE=${FEATS_TYPE:-1}
export LR=${LR:-0.001}
export SPLIT=${SPLIT:-1}
export REPEAT=${REPEAT:-5}

export CUDA_VISIBLE_DEVICES=1

python run_new.py  --dataset massreg \
--epoch=${EPOCH} \
--num-layers=${NUM_LAYERS} --num-heads=${NUM_HEADS} \
--hidden-dim=${N_HID} --dropout=${DROPOUT} \
--feats-type=${FEATS_TYPE} --lr=${LR} \
--chronological-split=${SPLIT} \
--repeat=${REPEAT}
