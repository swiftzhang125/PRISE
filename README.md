# PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment
Demo code for our proposed PRISE method on GoogleMap, GoogleEarth, and MSCOCO dataset.


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


