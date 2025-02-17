# ResFuSe

Decoder-Focused Knowledge Distillation for Lightweight U-Net Segmentation in Medical Imaging.

## Prerequisites

### 1.Prepare pre-trained ViT models

We use the teacher model given by [Beckschen/TransUNet: This repository includes the official project of TransUNet, presented in our paper: TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.](https://github.com/Beckschen/TransUNet) . If you need the teacher model, please contact me by email.

### 2.Prepare Dataset

#### Synapse Dataset :  

https://www.synapse.org/#!Synapse:syn3193805/wiki/217789.

#### ACDC Dataset: 

https://www.creatis.insa-lyon.fr/Challenge/acdc/

#### CC-CCII Dataset:

http://ncov-ai.big.ac.cn/download

Also, if you need processed data, please contact me by email.



## Framework

##### Training Commands

```python
python3 train.py --model transUnet --num_epochs 150 
```


```
python3 tester.py --model transUnet --num_epochs 150
```


```
python3 train_kd.py --model transUnet --num_epochs 150 --kd resFuSe
```


```
python3 tester.py --model transUnetStu --num_epochs 150 --kd resFuSe
```


The complete code will be updated after the paper is published...
