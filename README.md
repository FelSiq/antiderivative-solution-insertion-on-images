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
Here some examples of expected program inputs, at the left side, with the corresponding program output, at the right side.
<p>
  <img  src="/sample-inputs/sample-1.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-1.png" width="400" height="400" />
</p>
<p>
  <img  src="/sample-inputs/sample-2.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-2.png" width="400" height="400" />
</p>
<p>
  <img  src="/sample-inputs/sample-5.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-5.png" width="400" height="400" />
</p>
<p>
  <img  src="/sample-inputs/sample-6.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-6.png" width="400" height="400" />
</p>

This program is far from perfection. It needs much more training data to improve its performance. Here is some examples where things simply does not work fine.
<p>
  <img  src="/sample-inputs/sample-3.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-3.png" width="400" height="400" />
</p>
<p>
  <img  src="/sample-inputs/sample-4.jpg" width="400" height="400" />
  <img src="/sample-outputs/sample-4.png" width="400" height="400" />
</p>

# Workflow
![Image of the final workflow](/workflow.png)

The workflow is divided mainly in two parts:
* Symbol recognition with Convolutional Neural Network (CNN)
    1. Data must be balanced and preprocessed
    2. CNN architecture must be created
    3. Model must be trained
* Antiderivative recognition and insertion in-place
    1. Input must be preprocessed and segmented
    2. Objects must be detected and parsed
    3. Object recognition (uses the previous described part) and expression generation
    4. Solve the expression and insertion in-place

## Symbol recognition
Please check the [Symbol recognition subdirecotry README]("/symbol-recognition/README.md") for deeper information about the process symbol recognition process.

## Antiderivative recognition
To do.

### Insertion in-place
Wolfram Alpha already offers an image of the solution. This image is preprocessed, resized, and inserted in the original input image.
