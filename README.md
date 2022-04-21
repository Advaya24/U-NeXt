# U-NeXt
You can set up the environment using ```environment.yml```.
## Experiment 1: Aerial dataset

In order to set up the dataset for this, please use the code that downloads data for berlin and chicago https://github.com/alpemek/aerial-segmentation/blob/master/dataset/download_dataset.sh, and place them under ```aerial/dataset```

To run the U-NeXt model on this dataset, run:
```
cd aerial
python3 train_aerial.py --model UNext 
```
You can vary ```--model_type```, to change the model's size.

To run the baseline:
```
python3 train_aerial.py --model Resunet 
```

## Experiment 2: Synapse dataset
Dataset setup: please read the project_TransUNet/README.md (create ```project_TransUNet/data```)

To run the U-NeXt model:
```
cd project_TransUNet/TransUNet
python3 train.py --model UNext
```
You can vary ```--model_type```, to change the model's size.

For the baseline:
```
cd project_TransUNet/TransUNet
python3 train.py --model Resunet
```