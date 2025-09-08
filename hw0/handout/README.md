# Image Classifier for HW0

This project is an image classifier built with PyTorch. It's designed to solve the image classification task for HW0. The script can train two types of models: a simple fully-connected neural network and a convolutional neural network (CNN) to classify images into one of three classes.

## Features

*   **Two Model Architectures**: Choose between a simple multi-layer perceptron (`simple`) or a Convolutional Neural Network (`cnn`).
*   **Custom Data Loading**: Uses a custom PyTorch `Dataset` to load image paths and labels from CSV files.
*   **Data Preprocessing**: Includes standard image transformations like resizing, center cropping, and normalization. It also supports converting images to grayscale.
*   **Training and Evaluation**: Implements a standard training loop with evaluation on training, validation, and test sets after each epoch.
*   **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) for logging metrics (loss, accuracy) and visualizing images with their predicted and true labels.
*   **Configurable**: Easily configurable through command-line arguments for hyperparameters like learning rate, batch size, number of epochs, and model type.

## Installation

1.  **Install Miniconda:**
    If you don't have `conda` installed, follow the instructions here to install Miniconda:
    <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Create and activate a Conda environment:**
    The PDF suggests using Python 3.12.
    ```bash
    conda create -n hw0_env python=3.12
    conda activate hw0_env
    ```

4.  **Install PyTorch:**
    It is highly recommended to install PyTorch by following the official instructions for your specific system (OS, package manager, CUDA version). This will ensure you get the correct build for your hardware.
    Visit <https://pytorch.org/get-started/locally/> and run the command they provide.

5.  **Install other dependencies:**
    Once PyTorch is installed, you can install the rest of the dependencies using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    This file includes `wandb`, `pandas`, and other necessary packages. If you encounter any conflicts with the PyTorch version from `requirements.txt` and the one you installed manually, you can remove the `torch` and `torchvision` lines from `requirements.txt` before running the `pip install` command.

## Dataset

The script expects the dataset to be organized in a `data` directory at the root of the project. This directory should contain three CSV files:

*   `data/img_train.csv`: For training data.
*   `data/img_val.csv`: For validation data.
*   `data/img_test.csv`: For test data.

Each CSV file should have at least two columns:
*   `image`: The file path to the image.
*   `label_idx`: The integer index of the label for the image.

The label mapping used in the code is: `0: "parrot"`, `1: "narwhal"`, `2: "axolotl"`.

## Usage

You can run the script `img_classifier.py` from the terminal.

### Basic Training

To train the simple neural network for 5 epochs with a batch size of 8 and a learning rate of 0.001:

```bash
python img_classifier.py --model simple --n_epochs 5 --batch_size 8 --learning_rate 1e-3
```

To train the CNN model:

```bash
python img_classifier.py --model cnn --n_epochs 5 --batch_size 8 --learning_rate 1e-3
```

### Using Weights & Biases

To enable logging with Weights & Biases, you first need to be logged into your `wandb` account.

```bash
wandb login
```

Then, add the `--use_wandb` flag when running the script.

```bash
python img_classifier.py --model cnn --n_epochs 10 --use_wandb
```

To also log sample images with their predictions from the last epoch to `wandb`, use the `--log-images` flag.

```bash
python img_classifier.py --model cnn --n_epochs 10 --use_wandb --log-images
```

### Command-Line Arguments

Here is a full list of available command-line arguments:

*   `--use_wandb`: (flag) Use Weights and Biases for logging.
*   `--n_epochs`: (int) The number of training epochs. Default: `5`.
*   `--batch_size`: (int) The batch size for data loaders. Default: `8`.
*   `--learning_rate`: (float) The learning rate for the optimizer. Default: `1e-3`.
*   `--model`: (str) The model type to use. Choices: `['simple', 'cnn']`. Default: `'simple'`.
*   `--grayscale`: (flag) Use grayscale images instead of RGB.
*   `--log-images`: (flag) Log images to `wandb` at the end of training.

## Models

### Simple Neural Network

The `simple` model is a basic multi-layer perceptron with the following architecture:
1.  Flatten layer
2.  Linear layer (input features -> 512) with ReLU activation
3.  Linear layer (512 -> 512) with ReLU activation
4.  Linear layer (512 -> num_labels)

### CNN

The `cnn` model is a Convolutional Neural Network with a more complex architecture designed for image data. It consists of several convolutional layers, layer normalization, GELU activation, and an average pooling layer before the final linear layer for classification.

## Answering the Homework Questions

This codebase is designed to help you answer the questions in the "Image Classification" section of the homework PDF. Here's a guide on how to approach each question using this script.

### General Setup: Weights & Biases

Most questions require you to use Weights & Biases (`wandb`). Make sure you have an account and are logged in (`wandb login`). To enable logging for any script execution, add the `--use_wandb` flag. The script is already configured to log all the necessary metrics mentioned in the PDF, such as batch-level loss, epoch-level loss, and accuracy for all data splits.

### Question 3.1: Initial Run and Batch Loss

*   **Task**: Run the classifier with default settings and plot batch training loss vs. number of examples seen. Name the run "neural-the-narwhal".
*   **How-to**:
    1.  The script already logs batch loss and the number of training examples to `wandb`.
    2.  To name your run, you can edit the `wandb.init` call in `img_classifier.py`. The `name` parameter controls the run name. The PDF asks for the run to be named "neural-the-narwhal".
    3.  Run the script with default parameters and the `wandb` flag:
        ```bash
        python img_classifier.py --use_wandb
        ```
    4.  In your `wandb` dashboard, you can create the required plot.

### Question 3.2 & 3.3: Validation Metrics

*   **Task**: Add validation set evaluation and plot train/validation loss and accuracy per epoch.
*   **How-to**: The starter code already evaluates on the validation set (`val_dataloader`) at the end of each epoch and logs `val_accuracy_epoch` and `val_loss_epoch` to `wandb`. Simply run the script with `--use_wandb` and the plots can be generated in the `wandb` interface.

### Question 3.4: Grayscale Images

*   **Task**: Convert images to grayscale, adjust the model, and compare test accuracy.
*   **How-to**:
    1.  The script has a `--grayscale` flag that adds the grayscale transform.
    2.  **Important**: The starter code does **not** automatically adjust the model for single-channel images. As the PDF instructs, you must modify the code yourself. In `img_classifier.py`, the `NeuralNetwork` model's first linear layer is hardcoded to expect 3 channels (`num_channels`). You will need to modify the code to handle a single channel when the `--grayscale` flag is active. A simple way is to adjust the `num_channels` global variable.
    3.  Run the default model and the grayscale model and compare their test accuracies, which are printed to the console and logged to `wandb`.

### Question 3.5: Logging Images

*   **Task**: Log a batch of images from each dataset with their predicted and true labels on the last epoch.
*   **How-to**: The script has a `--log-images` flag that implements this exact functionality. Simply add this flag to your training command along with `--use_wandb`.
    ```bash
    python img_classifier.py --use_wandb --log-images
    ```

### Question 3.6: Using the CNN Model

*   **Task**: Implement and evaluate a new CNN model as described by the computation graph in the PDF.
*   **How-to**:
    1.  The `CNN` model described in the PDF is already implemented in `img_classifier.py` as the `CNN` class. You can use it by passing the `--model cnn` command-line argument.
    2.  To get the train/test accuracies for the original (`simple`) and new (`cnn`) models, run the script twice, changing the `--model` argument.
    3.  To calculate the number of parameters for each model, the script prints this information to the console at the beginning of each run. Look for the "Model size:" output.
    4.  For the plots, you can create two runs on `wandb` (e.g., "base-model" and "new-model" as suggested) by setting the `name` in `wandb.init` and changing the `--model` argument for each run. You can then compare them in the `wandb` UI.

### Text Classification Questions

The `txt_classifier.py` script is used for this section. Similar to the image classifier, it is configurable via command-line arguments.

### Question 4.1: Article Length Histogram

*   **Task**: Generate a histogram of the lengths of the news articles.
*   **How-to**: The script has a `--length_histogram` flag that will generate and save a histogram plot as `length_histogram_hw0_4_1.png`. Use the command specified in the PDF:
    ```bash
    python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram
    ```
    Setting `--max_len` to `-1` disables truncation and padding, giving you the raw article lengths.

### Question 4.2: Padding and Truncation

*   **Task**: Implement padding/truncation and observe the effect on runtime and batching.
*   **How-to**: The starter code in `txt_classifier.py` already contains the `TruncateToMaxLen` and `PadToMaxLen` classes. The `get_data` function automatically applies these transforms unless `--max_len` is less than 1.
    1.  Run the three commands listed in the PDF under question 4.2.a and 4.2.b.
    2.  The first two commands will run successfully, and you can record their runtimes.
    3.  The third command (`--batch_size 32 --max_len -1`) will produce an error, as expected. The error message is the answer to question 4.2.b.

### Question 4.3: Switching to Adam Optimizer

*   **Task**: Compare the performance of the SGD and Adam optimizers.
*   **How-to**: Use the `--optimizer` flag to switch between them. You will also need to set the learning rate (`--lr`) appropriately for each. The PDF suggests a learning rate of `5.0` for SGD (the default) and the default Adam learning rate, which is `0.001`.
    *   **SGD Run**:
        ```bash
        python txt_classifier.py --optimizer sgd --lr 5.0 --use_wandb
        ```
    *   **Adam Run**:
        ```bash
        python txt_classifier.py --optimizer adam --lr 0.001 --use_wandb
        ```
    The validation and test accuracies will be printed to the console and logged to `wandb`.

### Question 4.4: LSTM Model

*   **Task**: Implement and evaluate an LSTM-based text classifier.
*   **How-to**:
    1.  The starter code already includes an `LSTMTextClassifier` model. You can activate it with the `--model lstm` flag.
    2.  Run the original model (using Adam, as instructed) and the LSTM model.
        *   **Original Model (Adam)**:
            ```bash
            python txt_classifier.py --model simple --optimizer adam --lr 0.001 --use_wandb
            ```
        *   **LSTM Model (Adam)**:
            ```bash
            python txt_classifier.py --model lstm --optimizer adam --lr 0.001 --use_wandb
            ```
    3.  You can then use the results logged to `wandb` to compare their validation and test accuracies across epochs.
