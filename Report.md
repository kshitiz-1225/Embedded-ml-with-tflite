## Embedded Systems - Project
## Exploring TensorFlow Lite for Embedded Systems

### **Introduction**
This project explores the capabilities of deploying machine learning models on resource-constrained embedded devices using TensorFlow Lite (TFLite). The objectives were to optimize models for high accuracy, low latency, and minimal memory usage, and evaluate performance to determine the viability of edge AI applications.

### **About TfLite**

TensorFlow Lite is an open-source deep learning framework for on-device inference (edge computing). It provides tools to run trained models on mobile, embedded, and IoT devices. TensorFlow Lite enables machine learning on edge devices by allowing developers to deploy trained TensorFlow models to resource-constrained environments like smartphones and microcontrollers. The key capabilities include an API to perform inference and model optimization tools to quantize and compress models for faster execution and lower memory usage. TensorFlow Lite allows developers to bring machine learning to edge devices within their limitations.

### **Why Tflite?**
- TensorFlow Lite is a lightweight solution for mobile and embedded devices. It enables on-device machine learning inference with low latency and a small binary size.

### **How to run the code**
- CIFAR-10 Classification
    - To prepare the models, we run,
        ```
        $ python3 cifar10_classification.py
        ```
    - Once get the models ready, convert it to tflite using

        ```
        $ python3 convert_to_tflite_cifar.py
        ```
    - For Inferencing using the tflite
        ```
        $ python3 inferences_tflite_cifar.py
        ```
- Gesture Recognition
- To prepare the models, we run,
    ```
    $ python3 gesture_recognition.py
    ```
- Once get the models ready, convert it to tflite using

    ```
    $ python3 convert_to_tflite_gesture.py
    ```
- For Inferencing using the tflite
    ```
    $ python3 inferences_tflite_gesture.py
    ```
- Handwritten Digit Recognition
- Prepare the model followed by tflite inferencing, we run,
    ```
    $ python3 handwritten_digit_recognizer.py
    ```
**Code for the above files can be found at:** [Github Link](https://github.com/kshitiz-1225/Embedded-ml-with-tflite)

### **Tflite conversion pipeline**

![img](tflitepipeline.svg)

The TFLite conversion pipeline involves:
- Train and save TensorFlow model
- Convert model to TFLite format
- Evaluate TFLite model performance

**Key Steps**
1. Train TensorFlow model and save
2. Convert to TFLite using TFLite converter
3. Use TFLite model for inference
4. Compare TFLite accuracy to original TF model

### **Comparison among Tflite and original Model**

|Dataset|Original_model_size|Tflite_model_size|Original model accuracy_on_test_set|Tflite_model accuracy on test_set|
|---|---|---|---|---|
|cifar10|2.4GB|786MB|0.5496|0.5496|
|Gesture_recognition|434MB|145MB|0.8398|0.8398|
|Handwritten digit recognition|910 KB|237 KB|0.9935|0.9934|

### **Workflow**
- **CIFAR-10 Classification**
    - Image size: ($1 \times 32 \times 32 \times 3$)
    - Train images: 50000
    - Test images: 10000
    - Model created in TensorFlow
    - Training epochs: 10
    - Learning rate: $10^{-3}$
    - Optimizer: ADAM
    - Loss function: Sparse Categorical Cross-Entropy
    - Model saved for future processing (conversion to TFLite model)
- **Gesture Recognition**
    - Image size: ($1 \times 28 \times 28 \times 1$)
    - Train images: 27455
    - Test images: 7172
    - Model created in TensorFlow
    - Training epochs: 10
    - Learning rate: $10^{-3}$
    - Optimizer: ADAM
    - Loss function: Sparse Categorical Cross-Entropy
    - Model saved for future processing (conversion to TFLite model)
- **Handwritten Digit Recognizer**
    - Define and train a digit classifier model using TensorFlow.
    - Convert the trained TensorFlow model to TensorFlow Lite for deployment.
    - Import TensorFlow and other required libraries.
    - MNIST dataset: 60,000 training images and 10,000 testing images of handwritten digits.
    - Each MNIST image is a 28x28 grayscale image with a digit label (0 to 9).
    - Create and train a TensorFlow model using the Keras API on the MNIST train dataset.
    - The model processes 28px x 28px grayscale images, classifies them into 10 digit classes, evaluates its performance on the entire test dataset, and predicts labels based on the highest probabilities.
    - Convert the model to TensorFlow Lite format for mobile deployment.

    - Model Description
        - Shape of the training image : (60000, 28, 28)
        - In our model we  import the Sequential Model from Keras and add Conv2D, MaxPooling, Flatten, Dropout, and Dense layers.
        - Optimizers Use : Adam (Adaptive Moment Estimation)
        - Loss Function  : SparseCategoricalCrossentropy to compute the cross entropy loss between the labels and predictions
        - Model converted to TFLite format for mobile deployment.

### **Results and Observations**
- The initial models (cifar10_model, gesture_recognition_model, handwritten digit recognition model) experience significant size reduction after conversion to TFLite, resulting in a substantial decrease in model size.
- Interestingly, the accuracy remains nearly unchanged, with a negligible difference of only up to 6 or 7 decimal places.
- Consequently, these compact TFLite models are suitable for deployment on a wide range of resource-constrained platforms, including mobile devices, edge computing systems, IoT devices, and embedded systems, all while maintaining their accuracy.
- For the handwritten digit recognizer, the TensorFlow Lite model achieves an accuracy of 0.9935 on the test dataset without quantization and 0.9934 after quantization.
- Notably, the TFLite model is approximately 26% of the original model's size, with a minimal accuracy reduction of approximately 0.0001.

### **Acknowledgements**
- We are thankful to Dr. Binod Kumar for providing us opportunity to work on this interesting project.

---

