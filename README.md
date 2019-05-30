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
![Sample Input 4](/sample-inputs/sample-4.jpg){:height="50%" width="50%"}
![Sample Input 5](/sample-inputs/sample-5.jpg){:height="50%" width="50%"}
![Sample Input 1](/sample-inputs/sample-1.jpg){:height="50%" width="50%"}
![Sample Input 2](/sample-inputs/sample-2.jpg){:height="50%" width="50%"}
![Sample Input 3](/sample-inputs/sample-3.jpg){:height="50%" width="50%"}

# Partial report
Here is summarized all work done until now.

## Planned workflow
![Planned workflow here](/planned-workflow.png)

The planned workflow is divided mainly in two parts:
* Symbol recognition with Convolutional Neural Network (CNN)
    1. Data must be generated and preprocessed
    2. CNN architecture must be created
    3. Model must be trained
* Antiderivative recognition and insertion in-place
    1. Input must be preprocessed and segmented
    2. Objects must be detected and parsed
    3. Object recognition (uses the previous descibred part) and expression generation
    4. Solve the expression and insertion in-place

## Current progress
* The structure to generate the train data for object recognition is ready (see [symbol recognition](/symbol-recognition)) directory.
    1. It was created from 10 handwritten symbols for each 18 different objects (ranging from digits to mathematical operators), of a total of 180 handwritten symbols.
    2. After that data augmentation was used for each handwritten symbol to generate 10 extra variants using random combination of image rotation, shifts and zoom in/out within predefined intervals. This generates an augmented dataset with 1980 images (180 original handwritten images + 1800 randomly generated images)
    3. Then, each image from this data is preprocessed from RGB to grayscale, then segmented using global mean threshold and saved as a 32x32 image.
* Preprocessing of input images is ready (see [preprocess module](/antideriv/preprocess.py) module). Various alternatives were tested, and currently I'm using the following steps (note that it has high similarity applied to the preprocessing in the train data for object recognition):
    1. RGB to grayscale
    2. It applies Sobel filter for border detection 
    3. Then, the Otsu threshold is applied to segment the input image.
* Connection with Wolfram Alpha is ready and working.

## To-do
- Detect the coordinates of each segmented object
- Crop useless borders in input image to speed up the object detection process
- Create the CNN architecture
    - Also, train the CNN model
- Preprocess solution image and insert it inside the input image
