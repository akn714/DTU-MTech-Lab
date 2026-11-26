## Research Direction
```
My project focuses on **Low-Dimensional Discriminative Image Representation Learning**, which is a research area in Computer Vision and Machine Learning.

The main objective is to investigate how Convolutional Neural Networks (CNNs), sparse coding, and orthogonal dimensionality reduction can be integrated to learn compact and discriminative representations of images.

CNNs help extract meaningful visual features, while sparse coding identifies important patterns in the extracted features using a learned dictionary. Orthogonal projections further reduce the dimensionality of these features while preserving useful discriminative information. By integrating these techniques into an end-to-end learning framework, the approach aims to improve image representation and classification performance.

Unlike application-oriented projects such as disease detection, image enhancement, or object recognition, my project focuses on improving how computers extract, represent, and learn discriminative information from images.
```

## Tips to present the project
- Start with showing the paper, showing the publication it is published in and the published date.

## DIA Project PPT Essentials
- Code Snippet of every section/concept of the project
- Explain architecture of the project in detail
- Why a perticular concept or section is used in the architecture
- Check all the core numbers like:
    - how many kernals used (what is the size of the kernal)
    - how many number of layers
- Explain how and why the a formula is used
- Get complete ground information of the paper (like literature review etc.)
- Preprocessing and all the details of the dataset
- Explain all the things of the paper in detail

## Notes
#### Residual Networks (ResNet18)
- Youtube video: https://www.youtube.com/watch?v=k1LqFg768b8
- ResNet18 is trained on `ImageNet` dataset, and can only output classes among 1000 classes from ImageNet dataset.

#### Primary dataset: CIFAR-10
- Dataset Breakdown
    - Image format: 32×32 RGB color images (3 channels).
    - Splits: 5 training batches of 10,000 images each (50,000 total; 5,000 per class) and 1 test batch of 10,000 images (1,000 per class).
    - Classes (0–9): Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- Where to Access It
    - Download original versions and format specifications from [CIFAR-10 and CIFAR-100 datasets](https://cave.cs.toronto.edu/kriz/cifar.html).


#### Dimensionality Reduction
- PCA (Principal Component Analysis) is a dimensionality reduction technique.

- Sparse Coding: https://youtu.be/FE09pPNqUA8?si=qmQQaD_TAY6KKz4u
- PCA: https://youtu.be/Dv-Kk7PDEas?si=7VsNGZo0ipnd6YsO

- What is Sparse Dimensionality Reduction? It combines both concepts:
    - Reduce the original high-dimensional data into fewer dimensions.
    - Represent the reduced data using only a small number of non-zero coefficients.
    - Preserve the most discriminative information needed for classification.
- How does this relate to your IEEE paper? In this paper, the basic idea is to combine:
    - Convolution: Extract meaningful image features.
    - Sparse coding: Represent features using a small number of active dictionary elements (Active dictionary elements are the few learned patterns that the model actually uses to represent an image, eg. from 100 learned patterns, 5 are active or selected and rest 95 are inactive or zero).
    - Low-dimensional representation: Reduce feature dimensionality.
    - Discriminative learning: Ensure the learned representation helps distinguish different image classes.