<div align="center">
  <center><h1>Total Perspective vortex<br> Brain-Computer Interface (BCI) Project 🧠📈</h1></center>
  </div>

## Overview

This project is dedicated to developing a Brain-Computer Interface (BCI) leveraging advanced machine learning techniques and real-time data streaming. Key aspects include:

-   **Custom CSP Implementation:** A Common Spatial Patterns algorithm for effective EEG signal processing.
-   **Feature Extraction:** Extracting features from EEG data using Fast Fourier Transform (FFT).
-   **Ensemble Learning:** A combination of various machine learning classifiers such as XGBoost, Random Forest, Logistic Regression, and Linear Discriminant Analysis, enhancing prediction accuracy.
-   **Kafka Integration:** Utilizing Apache Kafka to establish a real-time data streaming platform, simulating live EEG data processing.

## Goal

Using Physionet EEG Motor Movement/Imagery Dataset, we aim to develop a BCI that can predict a subject's task based on EEG data.
The primary objective is to interpret EEG data to understand a subject's specific tasks or mental states, aiming to achieve high accuracy in real-time processing.

Subjects experiments are divided into 4 tasks. The goal is to predict the task number based on the EEG data.

Tasks are as follows:

1. Open and close left or right fist
2. Imagine opening and closing left or right fist
3. Open and close both fists or both feet
4. Imagine opening and closing both fists or both feet

## Data

The dataset is available on [Physionet](https://physionet.org/content/eegmmidb/1.0.0/).
Data are handled using [MNE](https://mne.tools/stable/index.html).

There are 109 subjects, each subject has 14 experimental runs (see [here](https://physionet.org/content/eegmmidb/1.0.0/) for more details).

## Implementation

The following steps are performed:

1. Data preprocessing (filtering, downsampling, epoching)
2. Feature extraction (FFT, CSP)
3. Hyperparameters tuning (XGBoost, Random Forest, Logistic Regression, Linear Discriminant Analysis)
4. Model selection with cross-validation and ensemble learning, best model is chosen based on accuracy (bucket of models method)
5. Model training and saving
6. Real-time prediction (Kafka integration with Docker)

## Usage

-   **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-   **Find the best model and train it for one task (3-14) on one subject (1-109):**

    ```bash
    python mybci.py --task=<task_number> --subject=<subject_number> train
    ```

-   **Stream predictions for one task (3-14) on one subject (1-109):**

    ```bash
    docker compose up -d
    python mybci.py --task=<task_number> --subject=<subject_number> predict
    ```

-   **Find the best models and train them on all tasks for all subjects (/!\ This will create 436 models):**

    ```bash
    python mybci.py train_all
    ```

-   **Print accuracy of saved models:**
    ```bash
    python mybci.py stats
    ```

-  **Stream predictions for all tasks on one subject (1-109):**
    ```bash
    docker compose up -d
    python mybci.py --subject=<subject_number> predict_all
    ```



## Examples

<p align="center">
<img src="https://raw.githubusercontent.com/ThePush/total-perspective-vortex/main/assets/train.png"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ThePush/total-perspective-vortex/main/assets/predict.png"/>
</p>