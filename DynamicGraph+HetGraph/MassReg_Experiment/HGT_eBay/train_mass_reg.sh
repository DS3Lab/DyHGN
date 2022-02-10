export CONV_NAME=${CONV_NAME:-'dysat-gcn'}
export MAX_EPOCHS=${MAX_EPOCHS:-2048}
export N_LAYERS=${N_LAYERS:-4}
export N_HEADS=${N_HEADS:-8}
export N_HID=${N_HID:-256}
export DROPOUT=${DROPOUT:-0.1}
export EMB_DIM=${EMB_DIM:-40}

export CUDA_VISIBLE_DEVICES=0

[[ "${GMV_FAIR}" -eq 1 ]] && { export FLAG_GMV_FAIR=--gmv-fair; } || { export FLAG_GMV_FAIR=; }

echo "GMV_FAIR_FLAG: ${FLAG_GMV_FAIR}"

for SEED in 0 1 2; do
    python train_mass_reg.py --conv-name=${CONV_NAME} --seed=${SEED} \
        --max-epochs=${MAX_EPOCHS} \
        --n-layers=${N_LAYERS} --n-heads=${N_HEADS}\
        --n-hid=${N_HID} --dropout=${DROPOUT} \
	--emb-dim=${EMB_DIM} \
        ${FLAG_GMV_FAIR} \
        || { echo "Error at seed $SEED!"; exit 1; }
done
