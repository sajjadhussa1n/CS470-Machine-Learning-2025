# CS470: Machine Learning (Spring 2025)

This repository contains **lecture materials, slides, quizzes, assignments, and practice notebooks** for the course **CS470 – Machine Learning (Spring 2025)**.

---

## Course Overview
This course provides a comprehensive introduction to the core principles and algorithms of **Machine Learning (ML)**, focusing on both **theory and practical implementation**.  
Students will learn how to build predictive models, evaluate performance, and understand how modern ML systems are trained and optimized.

---

## Topics Covered

1. **Introduction to Machine Learning**  
   - Traditional programming vs. ML approach  
   - Types of learning: Supervised, Unsupervised, Reinforcement  
   - Basic ML workflow and model evaluation  

2. **Linear Regression & Gradient Descent**  
   - Model representation and cost function  
   - Analytical and iterative optimization  
   - Implementation in Python  

3. **Regularization Techniques**  
   - Ridge, Lasso, and Elastic Net  
   - Bias–variance tradeoff  
   - Preventing overfitting  

4. **Logistic Regression and Softmax Regression**  
   - Binary and multi-class classification  
   - Cross-entropy loss  
   - Gradient computation and optimization  

5. **Precision, Recall, and ROC Curve (AUC-ROC)**  
   - Confusion matrix  
   - Precision–recall tradeoff  
   - ROC and AUC interpretation  

6. **Support Vector Machines (SVMs)**  
   - Maximum margin classifier  
   - Soft margin and kernel trick  
   - Non-linear decision boundaries  

7. **Decision Trees**  
   - Entropy and information gain  
   - Gini impurity  
   - Overfitting and pruning  

8. **Ensemble Learning and Random Forests**  
   - Bagging and boosting  
   - Random forests  
   - Gradient boosting methods  

9. **Unsupervised Learning (Clustering)**  
   - K-Means and K-Medoids  
   - Hierarchical clustering  
   - Evaluation of clustering  

10. **Dimensionality Reduction (PCA)**  
    - Eigen decomposition and covariance  
    - Feature extraction and visualization  
    - Applications in preprocessing  

11. **Introduction to Neural Networks**  
    - Neuron model and activation functions  
    - Network architecture  
    - Forward and backward propagation  

12. **Training Neural Networks**  
    - Loss functions and optimizers  
    - Vanishing gradients  
    - Regularization in deep learning  

13. **Introduction to Convolutional Neural Networks (CNNs)**  
    - Convolution, pooling, and feature maps  
    - Architecture overview (LeNet, AlexNet)  
    - Image classification case study  

---


---

## Getting Started

### 1. Clone the Repository

To get a local copy of the course materials:
```bash
git clone https://github.com/sajjadhussa1n/CS470-Machine-Learning-2025.git
cd CS470-Machine-Learning-2025
```

## 2. Updating Course Materials Weekly

New lectures, assignments, and notebooks will be uploaded to this repository each week.  
To ensure you always have the latest materials, **pull the updated version from the main branch** regularly.

### Step 1: Open Your Local Repository

If you have already cloned the repository, open your terminal or VS Code terminal and navigate to it:

```bash
cd CS470-Machine-Learning-2025
```
### Step 2: Pull the Latest Updates

Run the following command to fetch new lecture notes, slides, notebooks, and assignments:

```bash
git pull origin main
```
This command updates your local repository with the latest content from GitHub without deleting your existing files.

### Step 3: Handling Local Changes

If you’ve made edits (for example, while solving assignments) and encounter merge conflicts, you can temporarily save your local changes before pulling updates:

```bash
git stash
git pull origin main
git stash pop
```
---

## Tools & Environment
- **Programming Language:** Python 3.x  
- **Recommended IDEs:** VS Code, Jupyter Notebook, or Google Colab  
- **Core Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/PyTorch (for NN parts)

---

## License
All course materials in this repository — including lectures, slides, assignments, and notebooks — are provided for **academic and educational purposes**.

You are encouraged to **use, share, and adapt** the material for **personal learning, classroom teaching, or research**.

However, **commercial use, redistribution, or monetization** of any part of this repository **is strictly prohibited without prior written permission** from the instructor or course author.

---

## Instructor
**Dr. Sajjad Hussain**  
Department of Electrical and Computer Engineering,
School of Electrical Engineering and Computer Science (SEECS),
National University of Sciences and Technology (NUST), Pakistan.  
📧 *[sajjad.hussain2@seecs.edu.pk]*

---

*Based on material adapted from*  
- *A. Géron, “Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow” (3rd Edition)*  
- *Instructor’s custom slides and practical notebooks*
