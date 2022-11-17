# PRISE
Demo code for our proposed PRISE model on GoogleMap, GoogleEarth, and MSCOCO dataset.


## Requirements
Create a new anaconda environment and install all required packages before runing the code.
```bash
conda create --name prise
conda activate prise
pip install requirements.txt
```


## Dataset
You can follow the dataset preparation [here](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography). We will upload our dataset after our paper is published.

Please note that changing the data path in the ./src/data_read.py if necessary.


## Usage
To train a model to estimate the homography:
* Find a good initialization for the homography estimation
* Train the PRISE model
```bash
cd src
sh create_checkpoints.py
sh run.sh
```

To see the training loss and test reuslts under:
```bash
cd ./results/<dataset_name>/mu<mu>_rho<rho>_l<lambda_loss>_nsample<sample_noise>/trainig/
```
We will upload the pretrained model for each dataset after our paper is published.


## Advanced
To change the hyperparameters:
```bash
cd ./src/settings.py
```



