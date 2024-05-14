

# Deep Fake Video Detection

## Overview
Our Deep Fake Video Detection project leverages Convolutional Neural Networks (CNN) to accurately differentiate between real and synthetic videos. By achieving an accuracy of 77.82%, our model demonstrates a solid capability in identifying deep fake content, utilizing the comprehensive Celeb-DF dataset for training.

## Dataset
We trained our model using the Celeb-DeepFakeForensics dataset, a valuable resource for deep fake detection research, available at [this GitHub repository](https://github.com/yuezunli/celeb-deepfakeforensics). The dataset includes a wide range of real and manipulated videos, crucial for developing effective detection algorithms.



## Installation
Clone the repository and install the required dependencies to set up the project:


git clone https://github.com/lkasym/deepfake_project

pip install -r requirements.txt


## Usage
The project involves steps such as data preprocessing, model construction, training, and evaluation. Begin by preparing your data, then proceed to train the model with the dataset paths adjusted to your setup:


python deepfake.py


## Model Architecture
Our CNN model comprises several layers designed to process and analyze video frames effectively. This setup includes convolutional layers, max pooling, and dense layers, structured to capture and learn from the complexities of video data.

## Training and Evaluation
The model was trained with a focus on achieving a balanced representation of classes, using data augmentation to enhance the robustness of our training dataset. This approach helped us reach an accuracy of 77.82% on our test set, indicating the model's reliability in detecting deep fakes.

## Contributing
We welcome contributions from the community. Feel free to fork the repository, make improvements, and submit a pull request with your changes.





