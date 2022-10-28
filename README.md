Scar Severity Prediction
==========================

![image](https://user-images.githubusercontent.com/79613225/198250477-3924fb57-d3cd-4432-a36d-8740f1da47b3.png)


## Requirement
- Windows10, 3*RTX A4000, PyTorch 1.9.0, CUDA 11.2 + CuDNN 8.1.0, Python 3.9

## Usage
  
  
      # train
      python train_clinical.py --ngpu 3 --epochs 100 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --seed 1177 --prefix clinical_checkpoint ./[path_to_dataset]
      python train_image.py --ngpu 3 --epochs 100 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --kfold 5 --prefix image_checkpoint ./[path_to_dataset]
      python train_combined.py --ngpu 3 --epochs 100 --batch-size 16 --lr 0.01 --momentum 0.1 --weight-decay 1e-5 --seed 1177 --kfold 5 --prefix combined_checkpoint ./[path_to_dataset]
  

## Reference
- "CBAM: Convolutional Block Attention Module"
- Link: https://github.com/Jongchan/attention-module