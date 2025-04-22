# Intelligent Knowledge Reuse Models for Early-Onset Colorectal Cancer Detection and Lymph Node Metastasis Prediction Using CT Data

This repository contains code for the paper `Intelligent Knowledge Reuse Models for Early-Onset Colorectal Cancer Detection and Lymph Node Metastasis Prediction Using CT Data`.

## Code Structure

```bash
.
├── dataset.py          # load slice-level / patient-level CRC and LNM dataset  
├── lnm.py              # LNM train and eval  
├── model_explain.py    # bounding box and CAM visualization 
├── models
│   ├── config.py       # dataset config
│   ├── crc_model.py    # CRC Model 
│   ├── lnm_model.py    # LNM Model 
│   └── resnet3d.py
├── README.md
├── test.py             # CRC eval
├── train.py            # CRC train
└── utils   
    ├── data_utils.py  
    ├── losses.py
    ├── scheduler.py
    └── utils.py        
```

## Train 

Train Teacher and Student Models for CRC (MKDDN)

```bash
# teacher model
python train.py --model_name resnet --mode NAP --use_volume --loc --exp_name NAP_teacher_LO_loc_3D 
# student model
python train.py --epochs 15 --model_name resnet --mode N --use_volume --loc \
        --pretrain_weight_path run/ckpt/NAP_teacher_LO_loc_3D/best.pth --exp_name N_student_EO_loc_3D 
```

Train LNM prediction model (DBFFN)

```bash
# lnm dual model
python lnm.py --epochs 15 --batch_size 128 --lr 5e-5 --mode P --dual --exp_name PDual
```

## Test

Slice-level and patient-level performance for CRC

```bash
python test.py --model_name resnet --ckpt_path $CKPT_PATH --data_path $DATA_PATH --use_volume --loc --eval_individual_models 
```

Slice-level and patient-level performance for LNM

```bash
python lnm.py --mode P
```

Bounding box and cam visualization

```bash
# bounding box 
python model_explain.py --loc --vis --pretrain_weight_path run/ckpt/model_best.pth

# cam
python model_explain.py --vis --pretrain_weight_path run/ckpt/model_best.pth
```