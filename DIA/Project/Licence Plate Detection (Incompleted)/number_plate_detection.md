# Automatic Vehicle License Plate Detection and Recognition

## 1. Project Overview

The project aims to develop an Automatic Number Plate Recognition (ANPR) system that can detect a vehicle's license plate from an image/video and recognize the characters printed on it.

The project will combine **Digital Image Processing, Computer Vision, Object Detection, and OCR**.

The basic pipeline will be:

    Input Image
        ↓
    License Plate Detection
        ↓
    Plate Cropping
        ↓
    Image Preprocessing
        ↓
    Character Recognition (OCR)
        ↓
    Final License Plate Number

Example:

    Input → Car Image
             ↓
        [DL01AB1234]
             ↓
        OCR Processing
             ↓
        "DL01AB1234"

---

## 2. Main Idea

The project can compare two approaches for detecting the license plate:

### Approach 1 — Classical Image Processing

Use OpenCV techniques such as:

- Grayscale conversion
- Gaussian/Median filtering
- Canny edge detection
- Morphological operations
- Contour detection
- Aspect-ratio/geometric filtering

Pipeline:

    Image
      ↓
    Grayscale
      ↓
    Canny Edge Detection
      ↓
    Morphological Processing
      ↓
    Contour Detection
      ↓
    Candidate Plate Regions
      ↓
    Geometric Filtering
      ↓
    License Plate

### Approach 2 — Deep Learning

Train a YOLO model to detect the license plate directly.

    Image
      ↓
    YOLO
      ↓
    License Plate Bounding Box
      ↓
    Crop Plate

The final project can compare the classical approach with YOLO in terms of accuracy, speed and robustness.

---

## 3. OCR / Character Recognition

Once the plate is detected and cropped, preprocess it using:

- Grayscale
- Contrast enhancement
- CLAHE
- Thresholding
- Noise removal
- Optional perspective correction

Then use **Tesseract OCR** to recognize the characters.

Example:

    Plate Image
        ↓
    Preprocessing
        ↓
    Tesseract OCR
        ↓
    DL01AB1234

As an optional extension, individual characters can be segmented and classified using a CNN.

---

## 4. Datasets

### UFPR-ALPR

A good academic dataset for license plate detection and recognition.

It contains 4,500 annotated images and more than 30,000 license-plate characters.

Website:

https://web.inf.ufpr.br/vri/databases/ufpr-alpr/

The dataset contains Brazilian vehicles, so it can be used for general ALPR experimentation rather than specifically for Indian plates.

### CCPD

A large-scale Chinese license plate dataset containing 300,000+ images and challenging subsets involving blur, rotation, illumination and other conditions.

Repository:

https://github.com/detectRecog/CCPD

### Custom Indian Dataset

For the Indian-specific part of the project, create a small dataset of around **500-1000 vehicle images** and manually annotate the license plates.

Include:

- Different vehicle types
- Different viewing angles
- Different distances
- Day/night images
- Blurred images
- Different Indian states/plate formats

---

## 5. Simple Implementation Plan

### Step 1 — Literature Survey

Study:

- License plate detection
- Classical image processing
- YOLO object detection
- OCR

Select 1-2 manageable research papers as references.

### Step 2 — Classical Detector

Implement:

    Image
      ↓
    Grayscale
      ↓
    Canny
      ↓
    Morphology
      ↓
    Contours
      ↓
    Geometric Filtering
      ↓
    Plate

Test its performance on the dataset.

### Step 3 — YOLO Detector

Annotate license plates and train a small YOLO model.

Output:

    Bounding box + confidence

### Step 4 — Plate Preprocessing

After detection:

    Crop Plate
       ↓
    Resize
       ↓
    Grayscale
       ↓
    CLAHE / Contrast Enhancement
       ↓
    Thresholding
       ↓
    OCR

### Step 5 — OCR

Start with Tesseract.

If time permits, implement character segmentation + CNN and compare it with Tesseract.

### Step 6 — Experiments

Compare:

    Classical Detection vs YOLO

and test under:

- Normal images
- Blur
- Low resolution
- Different viewing angles
- Different illumination

### Step 7 — Evaluation

For detection:

- Precision
- Recall
- F1-score
- IoU
- mAP (for YOLO)

For recognition:

- Character accuracy
- Complete plate accuracy

---

## 6. Suggested Project Scope

The recommended semester-project scope is:

    Classical Plate Detection
             +
    YOLO Plate Detection
             +
    Image Preprocessing
             +
    Tesseract OCR
             +
    Performance Comparison

The CNN-based OCR and video/real-time ANPR can be treated as optional extensions.

---

## 7. Suggested Project Title

**Comparative Analysis of Classical and Deep Learning Approaches for Automatic Vehicle License Plate Detection and Recognition**

Short version:

**Automatic Vehicle License Plate Detection and Recognition**

---

## 8. Expected Final Demo

The final system should take an image and display:

    Original Image
          ↓
    Detected License Plate
          ↓
    Cropped/Enhanced Plate
          ↓
    Recognized Number

Example output:

    License Plate: DL01AB1234
    Detection Confidence: 94%
    OCR Result: DL01AB1234

A video/webcam version can be added if there is enough time.
