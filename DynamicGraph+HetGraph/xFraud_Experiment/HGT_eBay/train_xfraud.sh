export CONV_NAME=${CONV_NAME:-'dysat-gcn'}
export MAX_EPOCHS=${MAX_EPOCHS:-2048}
export N_LAYERS=${N_LAYERS:-2}
export N_HEADS=${N_HEADS:-4}
export N_HID=${N_HID:-128}
export DROPOUT=${DROPOUT:-0.1}
export EMB_DIM=${EMB_DIM:-10}
export TASK=${TASK:-'txn'}

export CUDA_VISIBLE_DEVICES=1

for SEED in 0 1 2; do
    python train_xfraud.py --conv-name=${CONV_NAME} --seed=${SEED} \
        --max-epochs=${MAX_EPOCHS} \
        --n-layers=${N_LAYERS} --n-heads=${N_HEADS}\
        --n-hid=${N_HID} --dropout=${DROPOUT} \
	--emb-dim=${EMB_DIM} --task=${TASK} \
        ${FLAG_GMV_FAIR} \
        || { echo "Error at seed $SEED!"; exit 1; }
done
