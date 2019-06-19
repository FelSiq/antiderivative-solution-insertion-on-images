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
<a link="workflow">

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
After the input image is highly preprocessed (see the [workflow schematic](#workflow)), it is scanned pixel by pixel searching for distinct objects. When an object is found, it is painted (following a algorithm pretty much like a Breadth First Search) and the coordinates of the slice of its boundaries (x and y maximal and minimal) collected.

Then, with the coordinates of each object window, each one is collected and resized to a 45x45 image, mantaining the aspect ratio as much as possible.The frozen CNN model is then feed with these separated objects, transforming they in a raw text expression.

This expression is send to Wolfram Alpha, which returns its solution as an image.

This image is also preprocessed, and finally inserted in the input image.

### Insertion in-place
Wolfram Alpha already offers an image of the solution. This image is preprocessed, resized, and inserted in the original input image.
