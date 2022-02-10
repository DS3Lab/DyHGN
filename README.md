# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Get started](#get-started)
  - [1. Env setup](#1-env-setup)
  - [2. DyHGN models](#2-dyhgn-models)
  - [3. Simple HGN](#3-simple-hgn)
  - [4. Graph derived features](#4-graph-derived-features)
- [5. Data files](#5-data-files)
- [6. Citation](#6-citation)
- [7. Reference](#7-reference)


# Introduction
Code for the paper submission: "Modelling graph dynamics in fraud detection with ``Attention"".

DyHGN (**Dy**namic **H**eterogeneous **G**raph Neural **N**etwork) to 
- (a1) detect suspicious account registration, 
- (a2) flag risky transactions,  
- (a3) identify risky accounts.

# Get started
## 1. Env setup
Setup the python environment with `conda` and install `pytorch` and its dependencies. We provide a requirements file with all the necessary packages. Notice for the `pytorch` related package, you may install the correct version to fit your `cuda` device. In the following experiment scripts, we will use the `cuda10.1` version.

## 2. DyHGN models
The DyHGN-* models mentionned in the paper can be found in the folder DynamicGraph+HetGraph. Since our datasets do not all have the same temporal granularity (weeks and days for MassReg and weeks only for xFraud), we distinguish 2 folders for the two types of datasets.

Now, if you want to run the DyHGN-DE model for instance on the MassReg dataset, you need to go to the right directory

```
cd DynamicGraph+HetGraph/MassReg_Experiment/HGT_eBay
```
Then you can either fill in the train_mass_reg.sh file with the right info and then run
```
./train_mass_reg.sh
```
OR you can directly run the following command
```
python train_mass_reg_DE.py --conv-name='dysat-gcn' --seed=$0  --max-epochs=2048 --n-layers=4 --n-hid=256 --dropout=0.1 --emb-dim=40
```

Notice that for the xFraud dataset you also need to specify the task parameter ('txn' for the transaction classification or 'account' for account classification).

## 3. Simple HGN

The Simple HGN folder is adapted from the github repo [HGB](https://github.com/THUDM/HGB) of the paper "Are we really making much progress? Revisiting, benchmarking and refining the Heterogeneous Graph Neural Networks." by Lv, Ding et al.

In order to run the experiments, you need to fill in the simple-hgn.sh file with your parameters. Then, run:
```
./simple-hgn.sh
```

## 4. Graph derived features
The notebook Handcrafted graph features refers to the Section 3.5 GNN vs. Models using Graph-derived Features of the paper.
Note this piece of code is under legal review and can be open-sourced after approval.


# 5. Data files

The data we use in the paper is proprietary, i.e., real-world transaction records on the eBay platform.

Note that eBay-small dataset (desensitized transaction records) in [1] is used in this work (denoted as xFraud). Please contact the authors (srao@ethz.ch) for DATA USE AND RESEARCH AGREEMENT (eBay) and obtain the usage rights of eBay-small dataset. In the long run, it would be possible to share the MassReg dataset after the legal review at eBay. We provide sample data under https://github.com/eBay/xFraud/tree/master/data. 

Since Simple-HGN requires another format of data, we also provides the right data to run the model in Simple-HGN/data for the three tasks at hand. Please contact authors for further information.

# 6. Citation
```text
@article{rao2022dyhgn,
title={Modelling graph dynamics in fraud detection with ``Attention"},
author={Rao, Susie Xi and Lanfranchi, Clémence and Zhang, Shuai and Han, Zhichao and Zhang, Zitao and Min, Wei and Cheng, Mo and Shan, Yinan and Zhao, Yang and Zhang, Ce},
journal={under review},
year={2022}
}
```
# 7. Reference
[1] Rao, Susie Xi, Shuai Zhang, Zhichao Han, Zitao Zhang, Wei Min, Zhiyao Chen, Yinan Shan, Yang Zhao, and Ce Zhang. "xFraud: explainable fraud transaction detection." Proceedings of the VLDB Endowment 15, no. 3 (2021): 427-436.

