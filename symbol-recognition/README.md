# Symbol recognition subdirectory

This subdirectory is dedicated to all structure related to symbol recognition. Its purpose is to preprocess the input data and create an adequate classifier model using Convolutional Neural Network (CNN).

- If you're interested in the steps taken until the construction of the used classifier model in the Antideriv module, continue reading this README.
- If you're interested in the used data in this subrepository, just download the "data-original.zip" file. This file is a subset of the original data whose source URL can be found in the [Data source](#data-source) section.

## Running
It is used Python 3. This module will not work with Python 2.

First, install this subdirectory required python packages.

```
pip install -Ur requirements
```

Then, you can execute all steps at once with the following command:

```
python runall.py
```

The [runall.py](./runall.py) script will execute all steps at once (described in the section [Data processing pipeline](#data-processing-pipeline)), in the correct order. You can also give to it some command line arguments to skip some steps.
1. "b": skip data dalancing
2. "a": skip data augmentation, and also the previous step
3. "p": skip data preprocessing, and also all previous steps

For example, the following code executes only the data preprocessing and model training steps.

```
python runall.py a
```

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

First, the mean of class sizes, T, is calculated. Then, a new dataset is created were every possible class have exactly T instances.

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
3. Padding of size 4 for all borders (filling with zeros the new pixels)
4. Binary dilation (morphology) with square mask of size length 3 (connectivity 2)
5. Local mean thresholding with 2x2 filter
6. All [preprocessing applied to an input image](/README.md#workflow.png)
7. Resize to 45x45 image using interpolation of third order and without anti-aliasing
8. Mean thresholding (once again), to remove resizing noise

The new data is placed in a new subrepository named "data-augmented-preprocessed."

## Predictive models using CNN
The preprocessed data is used to train CNN classifier models. It was tested 27 different CNN architectures for this task. All tested architectures were kept in the [Symbol recognition module](./symbol_recog.py) for reference.

Some models are chosen, and then re-trained with all available data (with validation splits to used early stopping).

Lastly, all trained chosen models are frozen and copied to the Antideriv module to form a predictive committee.

## Data source
I do not own the used datasets.

The data was retireved both from "[Handwritten math symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols)" and from "[HASYv2 - Handwritten Symbol database](https://zenodo.org/record/259444#.XQxXsiZrzeQ)."
