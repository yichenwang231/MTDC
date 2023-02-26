# Mutual-Taught Contrastive Clustering (MTCC)
This is the code for the paper "Mutual-Taught Contrastive Clustering"

# Environment
- python==3.8
- pytorch==1.11.0
- torchvision==0.12.0
- CUDA==11.3
- timm==0.6.12
- scikit-learn==1.1.3
- opencv-python==4.6.0.66
- pyyaml==6.0
- numpy==1.22.4
- munkres==1.1.4
# Usage
MTCC is composed of the training and boosting stages. Configurations such as model, dataset, temperature, etc. could be set with argparse. Clustering performance is evaluated during the training or boosting.
# Data Preparation
CIFAR-10 could be automatically downloaded by Pytorch.Other datasets can be downloaded from the url provided by their corresponding papers or official websites.
# Dataset Structure:
Make sure to put the files in the following structure:
```
|-- datasets
|   |-- RSOD
|   |-- UC-Merced
|   |-- ...
```
# Training
After setting the configuration, to start training, simply run
```
python train.py 
```
# Boosting
Once the training is completed, there will be a saved model in the "model_path" specified in the configuration file. To boost the trained model, run
```
python boost.py 
```
# Test
To test the training or boosting model, run
```
python cluster.py 
```
# Acknowledge
MTCC is developed based on the architecture of "Contrastive Clustering" (AAAI 2021, https://github.com/Yunfan-Li/Contrastive-Clustering) and "Vision Transformer for Contrastive Clustering" (CVPR 2022, https://github.com/JackKoLing/VTCC) .We sincerely thank the authors for the excellent works!
