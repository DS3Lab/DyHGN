
export GMV_FAIR=0   # [0, 1]
export DROPOUT=0.1  # [0.1, 0.25, 0.5]
export N_LAYERS=8   # [4, 8, 12]

export CUDA_VISIBLE_DEVICES=3
export CONV_NAME='dysat-gcn'

export CUDA_VISIBLE_DEVICES=1
export CONV_NAME='gcn'


export CUDA_VISIBLE_DEVICES=2
export CONV_NAME='gat'

export CUDA_VISIBLE_DEVICES=0
export CONV_NAME=""


. activate eth
bash train_mass_reg.sh
