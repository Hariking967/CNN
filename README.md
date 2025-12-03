# 📘 NumPy Convolutional Neural Network (CNN) From Scratch

A fully handcrafted **Convolutional Neural Network** for MNIST digit recognition, built using **only NumPy and Python’s math module** —  
**no PyTorch, no TensorFlow, no external ML libraries.**

This project implements:

- ✔️ Custom convolution layers  
- ✔️ Custom ReLU  
- ✔️ Custom max-pooling  
- ✔️ Custom flattening  
- ✔️ Custom fully connected layers  
- ✔️ Custom forward propagation  
- ✔️ Custom cross-entropy loss (logits version)  
- ✔️ Numerical gradient descent (finite difference method)  
- ✔️ 100% manual implementation of everything  

This project is intended as a **mathematical + educational reproduction of deep learning mechanics**, not a training-optimized CNN.

---

## 🌟 Motivation

The goal of this project is to understand **every internal detail** of a CNN:

- How convolution works at the pixel level  
- How pooling compresses features  
- How logits become class scores  
- How loss is computed mathematically  
- How weights are updated using numerical gradients  

This is how neural networks were first implemented in academic research before modern frameworks existed.

---

## 🚀 Features

### 🔧 1. Convolution Layer (from scratch)
- Sliding windows over images  
- Multi-channel convolutions  
- Learnable filters  
- Manual patch extraction  
- Fully nested loops (no shortcuts)

### ⚡ 2. Max-Pooling (from scratch)
- 2×2 pooling  
- Fully manual selection of max values  
- No library tricks  

### 🧠 3. Forward Propagation  
The full forward pass is computed manually:
Image → Conv1 → Conv2 → MaxPool → Flatten → FC → Logits

Gradients computed using:

\[
\frac{L(W + h) - L(W)}{h}
\]

This is the mathematically pure way to approximate derivatives.

### 🔒 5. Zero external ML libraries  
Only:

- `numpy`  
- `math`  
- `torchvision` (for data loading only, not ML)

---
## 📊 Dataset

This project uses **MNIST Handwritten Digits**:

- **60,000 training images**  
- **10,000 testing images**  
- Grayscale (1 channel)  
- Resolution: 28×28  

Images are normalized and zero-padded to 30×30.

---

## 🧮 Mathematical Correctness

This project implements:

- ✔️ Logits  
- ✔️ Softmax (implicitly inside cross-entropy)  
- ✔️ Cross-entropy loss  
- ✔️ Numerical gradient estimation  
- ✔️ Weight update rule:

\[
W = W - \eta \cdot \frac{\partial L}{\partial W}
\]

Everything is **mathematically valid, exact, and correct**.

This behaves exactly as expected for a numerical gradient checker.

---

## ⚠️ Performance Notice

This implementation uses **finite differences** for gradient estimation.

This means:

- Extremely slow  
- Not meant for real training  
- Intended for learning and understanding  

This is **NOT** a high-performance CNN —  
This is an **educational deep-learning engine** built from scratch.

---

## 🏆 Why This Project is Special

- No copying  
- No shortcuts  
- No high-level libraries  
- 100% conceptual understanding  
- Everything written manually  
- Demonstrates true ML intuition

---

## 📌 Future Work

You can extend this project by adding:

- Backpropagation (analytic gradients)  
- Batch training  
- Better activation functions  
- Regularization  
- Momentum or Adam  
- More conv layers  
- Visualization of filters  

---

## 🙌 Author

Built entirely from scratch by **Sri Hari S**,  
using only:

- NumPy  
- math  
- deep understanding  
- curiosity  
- and pure logic.
