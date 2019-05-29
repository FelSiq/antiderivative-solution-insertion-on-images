# Student data
- Name: Felipe Alves Siqueira
- NoUSP: 9847706

# Title
Detection and solution of antiderivatives and insertion in-place in photos using image processing techniques.

# Abstract
The premise is to identify antiderivatives (because proper integrals seems to be harder) from photos, parse it, and insert its solution inside the image (preferably at the end of the detected object.) I'll just focus in supporting just some mathematical operations (mostly arithmetic) and, of course, also leave the mathematical solution to Wolfram Alpha. My focus is on the detection of the antiderivatives itself and the recognition of its operations in the correct order.

# Expected image processing technologies
- Object detection (for the antiderivative expression itself)
- Digit and Operator recognition

# Sample inputs
Here some examples of expected program inputs.
![Sample Input 4|50%](/sample-inputs/sample-4.jpg)
![Sample Input 5|50%](/sample-inputs/sample-5.jpg)
![Sample Input 1|50%](/sample-inputs/sample-1.jpg)
![Sample Input 2|50%](/sample-inputs/sample-2.jpg)
![Sample Input 3|50%](/sample-inputs/sample-3.jpg)

# Partial report
Here is summarized all work done until now.

## Current progress
* The structure to generate the train data for object recognition is ready (see [symbol recognition]("/symbol-recognition")) directory.
    1. It was created from 10 32x32 handwritten symbols.
    2. After that data augmentation is used for each handwritten symbol to generate 10 extra variants using random rotation, shifts and zoom in/out within predefined intervals.
    3. Then, this data is preprocessed using threshold to binarize each image.
* Preprocessing of input images is ready (see [preprocess module]("/antideriv/preprocess.py") module). Various alternatives were tested, and currently I'm using the following steps:
    1. RGB to grayscale
    2. It applies Sobel filter for border detection 
    3. Then, the Otsu threshold is applied to segment the input image.
* Connection with Wolfram Alpha is ready.

## To-do
- Detect the coordinates of each segmented object
- Create the CNN architecture
    - Also, train CNN model
- Preprocess solution image and insert inside the input image
