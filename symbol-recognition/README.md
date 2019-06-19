# Symbol recognition subdirectory

This subdirectory is dedicated to all structure related to symbol recoginition. Its purpose is to preprocess the input data and create an adequate classifier model using Convolutional Neural Network (CNN).

- If you're interested in the steps taken until the construction of the used classifier model in the Antideriv module, continue reading this README.
- If you're interested in the used data in this subrepository, just download the "data-original.zip" file. This file is a subset of the original data whose source URL can be found in the [Data source](#data-source) section.

## Running
It is used Python 3. This module will not work with Python 2.

First, install this subdirectory required python packages.
`
pip install -Ur requirements
`

Then, you can execute all step at once with the simple command just below.
`
python runall.py
`

The [runall.py](./runall.py) script will execute all steps at once (described in the section [Data processing pipeline](#data-processing-pipeline)), in the correct order.

You can also run each script separately, if the supposed script order is kept correctly.

## Data processing pipeline
<a link="data-processing-pipeline">

There are three main modules related to the data processing in this subdirectory. They are listed below, and executed in exactly the same order. You can click on each step to go to the related code.
1. [Data balancing](./balancing.py)
2. [Data augmentation](./augmentation.py)
3. [Data preprocessing](./preprocessing.py)

These scripts must be executed in exactly this order to everything works fine.

### Data balancing:
The original dataset (data-original.zip) was highly unbalanced. For that matter, it is balanced before anything else. The strategy adopted combines both undersampling and oversampling techniques.

First, the trimmed mean (cut-off of 10% in both extremes) of class sizes, T, is calculated. Then, a new dataset is created were every possible class have exactly T instances.

Obviously, all classes that has more than T instances are undersampled (T instances are picked randomly and uniformly), and all classes which size is under T are oversampled (again, instances are chosen randomly and uniformly to be replicated).

The new dataset is placed in a subrepository named "data-balanced."

### Data augmentation
In this step, each instance from the balanced dataset is modified exactly twice, to generate two new instances.

These modifications consists in random small pixel shifts, rotations and zooms.

Then, all three instances (the original and the new two instances) are placed in a new subrepository named "data-augmented."

This process is repeated for every single instance of the balanced dataset, effectively generating a new dataset whose size is the triple of the size of the "data-balanced" dataset.

### Data preprocessing
Finally, is this step, every instance in the "data-augmented" dataset is preprocessed. The exactly steps of the preprocessing (repeated for every single instance) are listed below (necessarily in this order):

1. RGB to Grayscale
2. Mean thresholding
3. Binary dilation (morphology)
4. Empty border crop
5. Resize to 45x45 image using interpolation of third order and without anti-aliasing
6. Mean thresholding (once again)

The new data is placed in a new subrepository named "data-augmented-preprocessed."

## Predictive models using CNN
The preprocessed data is used to train a CNN classifier model. It was tested 24 different CNN architectures for this task. All tested architectures were kept in the [Symbol recognition module](./symbol_recog.py) for reference.

The best model (with higher evaluation accuracy) is chosen, and then re-trained with all available data (with validation splits to used early stopping).

Lastly, the trained chosen model is frozen and copied to the Antideriv module.

## Data source
I do not own the used dataset. The data was retireved from "[Handwritten math symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols)."
