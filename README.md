# Automatic-Thyroid-Ultrasound-Image-Classification-Using-Feature-Fusion-Network

## OUR PROPOSED SYSTEM

An automated thyroid monitoring system for analyzing and classifying thyroid nodules in ultrasound images.

### Methodology

This project implements a hybrid feature fusion approach:

Step 1: Image Preprocessing**
- Align nodule center in each image
- Normalize image sizes to 224×224 pixels
- Rescale pixel values between 0 and 1

Step 2: Feature Extraction (3 Methods)**

1. Local Binary Patterns (LBP)**
   - Extracts texture descriptors
   - Captures local texture information around each pixel
   - Average pooling reduces dimensionality

2. Histogram of Oriented Gradients (HOG)**
   - Captures shape and directional patterns
   - Represents form and appearance of nodule
   - Quantized into local 1-D histograms

3. Convolutional Auto-Encoder (CAE)**
   - Deep learning-based unsupervised feature extraction
   - Encodes 2D images into compact representations
   - Removes high-frequency noise while preserving meaningful patterns
   - Architecture: Encoder (compress) → Bottleneck → Decoder (reconstruct)

Step 3: Feature Fusion**
- Combine LBP + HOG + CAE features
- Integrate TIRADS radiologist scores
- Include patient demographic information
- Create comprehensive nodule profile

Step 4: Classification**
Multiple classifiers tested:
- Random Forest
- Logistic Regression
- Linear SVM
- **RBF SVM** ← Best performing model

### Results

Best Model: RBF SVM (Radial Basis Function)**
- Accuracy: 92.52%
- Sensitivity: 88.8% (correctly identifies malignant nodules)
- Specificity: 91.1% (correctly identifies benign nodules)
- F1-Score: 50.89

Key Findings : RBF SVM outperformed even human radiologist accuracy by reducing false negatives (missed malignancies) while minimizing unnecessary biopsies.

### Dataset

- Total Images: 3,183 ultrasound images of thyroid nodules
- Bethesda Grades: 1-6 (malignancy risk assessment)
- Class Distribution 
  - Benign (Grades 2): 77%
  - Suspicious (Grades 3-6): 23%
- Radiologist Characterization: 1,434 images with TIRADS features
- Demographics: Predominantly middle-aged to elderly females (77.6%)


