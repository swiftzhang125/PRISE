# PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment
Demo code for our proposed PRISE method on GoogleMap, GoogleEarth, and MSCOCO dataset.

## Introduction
We propose PRISE to enforce the neural network to approximately learn a star-convex loss landscape around the ground truth give any data to facilitate the convergence of the LK method to the ground truth through the high dimensional space defined by the network.

<div align=center><img src="https://github.com/swiftzhang125/PRISE/blob/main/image/fig1.png" width="500" height="350" alt="Compared with DeepLK"/></div>



## Requirements
Create a new anaconda environment and install all required packages before runing the code.
```bash
conda create --name prise
conda activate prise
pip install requirements.txt
```


## Dataset
You can follow the dataset preparation [here](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography). 

Please note that changing the data path if necessary.
```bash
./src/ # modfiy the data_read.py
```


## Usage
To train a model to estimate the homography:
* Step1: Finding a good initialization for the homography estimation
* Step2: Train the PRISE model
```bash
cd src
sh create_checkpoints.py # step1
sh run.sh # step2
```

To see the training loss and test reuslts under:
```bash
cd ./results/<dataset_name>/mu<mu>_rho<rho>_l<lambda_loss>_nsample<sample_noise>/trainig/
```

## Performance
| Method  | PE < 0.1 | PE < 0.5 | PE < 1| PE < 3| PE < 5 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SIFT + RANSAC  | 0.00 | 4.70 | 68.32 | 84.21 | 90.32 | 95.26 | 96.55 |
| SIFT + MAGSAC  | 0.00 | 3.66 | 76.27 | 93.26 | 94.22 | 95.32 | 97.26 |


## Advanced
To change the hyperparameters:
```bash
cd ./src/ # and modify the settings.py
```
If you are looking for Pytorch implementation of our Star-Convex Constraints
```bash
cd ./py-sc/
```

## Publication
Please cite our papers if you use our idea or codes:
```bash
@inproceedings{zhang2023prise,
  title={PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment},
  author={Zhang, Yiqing and Xinming, Huang and Zhang, Ziming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```


