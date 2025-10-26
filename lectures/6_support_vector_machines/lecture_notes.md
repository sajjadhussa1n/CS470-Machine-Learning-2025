# Week 06 - Support Vector Machines (SVM)

This week, we will develop both an intuitive and mathematical understanding of Support Vector Machines (SVMs) — one of the most powerful and versatile algorithms in machine learning. 

---

## 1. Introduction to Support Vector Machines

Support Vector Machines (SVMs) are supervised learning models that find the best possible boundary between classes. Unlike simple classifiers that just separate data, SVMs aim to find the optimal separating hyperplane — the one that maximizes the distance (called the margin) between itself and the nearest data points from each class.

The **intuition** is straightforward: among all possible boundaries that separate the classes, we want the one that keeps the largest **safety buffer** between the two. This margin-based approach gives SVMs strong generalization capabilities.

<img src="./images/svm_basic.png" alt="Maximum_Margin_Classifier" width="500"/>
Figure: Example of Binary Classification using Mamimum Margin

## 2. Why Maximize the Margin?

Imagine driving along a wide road versus a narrow alley. On the wide road, there’s less chance of bumping into obstacles, even if you slightly drift. Similarly, an SVM aims to position the decision boundary as far away as possible from any data point of either class. This buffer zone — **the margin** — acts as a confidence region.

A larger margin in a Support Vector Machine generally leads to more robust and reliable predictions, as the model becomes less sensitive to small variations or noise in the training data. This wider separation between classes tends to improve the model’s ability to perform well on unseen data, enhancing its generalization capability. By maintaining a larger margin, the risk of overfitting is reduced because the model focuses on the overall structure of the data rather than trying to perfectly classify every training point. On the other hand, if the decision boundary is positioned too close to the data points, it may fit the training examples very accurately but fail to generalize to new samples, resulting in poor performance on test data.

## 3. Hard Margin Classification

In an ideal scenario where data is perfectly linearly separable, we can draw a straight line (in 2D) or a hyperplane (in higher dimensions) that cleanly separates the two classes. This is known as the **Hard Margin SVM**.

The goal is to find the hyperplane that maximizes the margin while ensuring every point is correctly classified. 

Mathematically, if we denote a hyperplane by its weight vector w and bias b, the constraints for correctly classified data are:

```math
y_i (w^\top x_i + b) \ge 1 \quad \forall i
```
We want to maximize the margin, which is equivalent to minimizing the norm of the weight vector:

```math
\min_{w, b} \ \frac{1}{2} \| w \|^2
```

Points that lie exactly on the margin boundaries are called **support vectors** — they are the “critical” points that define the decision boundary. Removing any non-support vector does not affect the hyperplane.

![Basic linear separation example](./images/support_vectors.png)
*Figure: Support Vectors defining the Margin*