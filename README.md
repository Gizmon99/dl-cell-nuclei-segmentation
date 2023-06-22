# dl-cell-nuclei-segmentation

## Overview

### Team:
Grzegorz Maliniak, Piotr Słowik
### Description:
Cell nuclei segmentation with Mask R-CNN
### Dataset:
Fluorescence Image Dataset: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265
![Screenshot](assets/sample_images_targets.png)

## How to use

1. Download the repository by cloning it or downloading the ZIP file and extract it to a folder of your choice. Open terminal in repos folder.

2. Create a virtual environment and activate it. This can be done by running the following commands in the terminal:
```
python -m venv env
source env/bin/activate
```

3. Install the project dependencies by running the following command in the terminal:
```
pip install -r src/requirements.txt
```

4. Download the dataset into the `data` directory by running the following command in the terminal:
```
dvc pull
```

5. Preprocess the images and create a Torch DataLoader for the dataset by running the following command in the terminal:
```
kedro run --pipeline='data_loader'
```
This command will also split the data into training and testing datasets and create training weights.

6. Create the Mask R-CNN model by running the following command in the terminal:

```
kedro run --pipeline='mask_rcnn_setup'
```

8. Train the Mask R-CNN model. The trained model will be saved inside Kedro.

```
kedro run --pipeline='model_train'
```

9. Evaluate the Mask R-CNN model and check it's performance.

```
kedro run --pipeline='model_evaluate'
```

10. To create documentation, use the following command in the terminal:
```
kedro build-docs
```


### Short version

1. Download the repository by cloning it or downloading the ZIP file and extract it to a folder of your choice. Open terminal in repos folder.

2. Type in terminal:

```
python -m venv env
source env/bin/activate
pip install -r src/requirements.txt
dvc pull
kedro run
```
