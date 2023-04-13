# SANF-AD
ICME2023--A Semantic-awareness Normalizing Flow Model for Anomaly Detection. 
Accepted.


## Paper Introduction
Anomaly detection in computer vision aims to detect outliers from input image data. Examples include texture defect detection and semantic discrepancy detection. However, existing methods are limited in detecting both types of anomalies, especially for the latter. In this work, we propose a novel semantics-aware normalizing flow model to address the above challenges. First, we employ the semantic features extracted from a backbone network as the initial input of the normalizing flow model, which learns the mapping from the normal data to a normal distribution according to semantic attributes, thus enhances the discrimination of semantic anomaly detection. Second, we design a new feature fusion module in the normalizing flow model to integrate texture features and semantic features, which can substantially improve the fitting of the distribution function with input data, thus achieving improved performance for the detection of both types of anomalies. Extensive experiments on five well-known datasets for semantic anomaly detection show that the proposed method outperforms the state-of-the-art baselines.


## Installation

1. First clone the repository
   ```
   git clone https://github.com/SYLan2019/SANF-AD.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n SANF-AD python=3.8
    ```
3. Activate the virtual environment.
    ```
    conda activate SANF-AD
    ```
4. Install the dependencies.
   ```
   pip install --user --requirement requirements.txt
   ```
   
## Configure and Run
All configurations concerning data, model, training etc, can be found in _config.py_. The default configuration will run a training with paper-given parameters on the Cifar10 dataset.

To replicate the results in the paper for CIFAR10  dataset, please set the paramater pretrained = False in _config.py_ and run the following commands to extract features:

``` shell
# CIFAR
python extractor.py
```

## Training
Then run the following code to train the model on the features of Cifar10
```
python train.py -h
```

### Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:
``` 
python main.py 
```

