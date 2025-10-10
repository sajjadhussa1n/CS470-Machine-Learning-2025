# CS470: Machine Learning  
## Week 01 — Introduction to Machine Learning  

---

### 1. Introduction and Motivation

Machine Learning (ML) is one of the most transformative fields in computer science today. It allows computers to **learn patterns from data** and **make predictions or decisions** without being explicitly programmed for each possible situation.

Think of everyday examples:
- Your email automatically categorizes messages as spam or not.
- Streaming platforms like Netflix or Spotify recommend movies or songs.
- Banks detect fraudulent transactions in real-time.
- Self-driving cars identify pedestrians and traffic signs from camera feeds.

All of these are powered by machine learning algorithms trained to recognize complex relationships between inputs and outputs.

At its core, ML is about building systems that **improve with experience** — much like humans. The more data they see, the better they perform.

---

### 2. What Is Machine Learning?

Tom Mitchell (1997) gave one of the most widely accepted definitions:

> **A computer program is said to learn from experience (E) with respect to some class of tasks (T) and performance measure (P), if its performance at tasks in T, as measured by P, improves with experience E.**

Let’s break it down:
- **Task (T):** What the system is trying to do (e.g., classify emails as spam/not spam).
- **Experience (E):** The data the system learns from (e.g., thousands of labeled emails).
- **Performance (P):** How success is measured (e.g., accuracy or F1-score).

A spam filter, for example, improves its ability to detect spam the more emails it processes and learns from.

---

### 3. Why Machine Learning?

Traditionally, programmers write explicit rules to solve a problem. But for many tasks — like recognizing speech or handwriting — **rules are too complex or numerous to define manually**. ML automates this process by learning from examples instead.

| Traditional Programming | Machine Learning |
|--------------------------|------------------|
| Programmer defines the rules | Model learns rules from data |
| Deterministic behavior | Probabilistic, data-driven behavior |
| Example: Sorting numbers | Example: Image recognition |

In short, ML shifts the paradigm from **rule-based programming** to **data-driven learning**.

---

### 4. Types of Machine Learning

Machine Learning can be broadly categorized based on the type of feedback available from the data.

#### 4.1 Supervised Learning

In **supervised learning**, we have both input features (**X**) and target outputs (**Y**).  
The model learns a mapping function:  
\[
f: X \rightarrow Y
\]
and is evaluated by how accurately it predicts \( Y \) for unseen \( X \).

**Examples:**
- Predicting house prices (Regression)
- Email spam detection (Classification)

**Algorithms:** Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines, Neural Networks.

#### 4.2 Unsupervised Learning

Here, only the inputs \( X \) are available — no labeled outputs.  
The goal is to **discover hidden structures or patterns** within the data.

**Examples:**
- Grouping customers by purchase behavior (Clustering)
- Reducing dimensionality for visualization (PCA)

**Algorithms:** K-Means, DBSCAN, PCA, Autoencoders.

#### 4.3 Semi-Supervised Learning

A combination of labeled and unlabeled data. Often used when labeling is expensive but large amounts of unlabeled data are available.

**Example:** Training a face recognition model using a small set of labeled faces and many unlabeled ones.

#### 4.4 Reinforcement Learning

In this paradigm, an **agent** learns by interacting with an environment.  
It takes actions, receives rewards or penalties, and gradually learns a policy that maximizes cumulative rewards.

**Examples:**
- Game playing (AlphaGo)
- Robotics control
- Traffic light optimization

---

### 5. Key Terminology

Understanding some fundamental terms is crucial before diving deeper.

| Term | Meaning |
|------|----------|
| **Instance / Example** | A single data point (e.g., one email or one image) |
| **Feature / Attribute** | A measurable property of an instance (e.g., word frequency, pixel value) |
| **Label / Target** | The outcome we want to predict |
| **Training Set** | Data used to train the model |
| **Test Set** | Data used to evaluate the model |
| **Model Parameters** | The internal values (weights) the algorithm learns |
| **Hyperparameters** | External settings chosen before training (e.g., learning rate, number of trees) |

---

### 6. The Machine Learning Pipeline

The overall workflow of an ML project generally follows these steps:

1. **Data Collection:** Gather data from sensors, databases, or APIs.
2. **Data Cleaning & Preprocessing:** Handle missing values, normalize scales, and encode categorical variables.
3. **Feature Engineering:** Select or create relevant features to represent the problem.
4. **Model Selection:** Choose an appropriate algorithm (e.g., linear regression, neural network).
5. **Training:** Optimize model parameters to minimize a loss function.
6. **Evaluation:** Assess performance using metrics like accuracy, precision, or RMSE.
7. **Deployment:** Integrate the trained model into an application or system.
8. **Monitoring:** Track model performance and retrain as needed.

---

### 7. Training and Testing Paradigm

A fundamental principle in ML is to **train on one set of data** and **test on another**.

- **Training Set:** Used to learn model parameters.
- **Validation Set:** Used to tune hyperparameters and prevent overfitting.
- **Test Set:** Used only once, to evaluate generalization performance.

This separation ensures that the model does not “memorize” the training data and can perform well on unseen examples.

---

### 8. Underfitting and Overfitting

A model should strike a balance between **simplicity** and **complexity**.

- **Underfitting:** The model is too simple to capture the underlying trend (e.g., a straight line fitted to a nonlinear curve).  
- **Overfitting:** The model is too complex and fits even the noise in training data, failing to generalize.

Visual intuition:

![Overfitting and Underfitting](../../assets/overfit_underfit.png)

To handle this:
- Use **regularization** to penalize complexity.
- Apply **cross-validation**.
- Gather **more training data**.
- Simplify the model or reduce the number of features.

---

### 9. Evaluating Performance

Depending on the problem type:

- **Regression Tasks:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), \( R^2 \)
- **Classification Tasks:** Accuracy, Precision, Recall, F1-score, ROC-AUC

Each metric reveals different aspects of model performance. We will study these in detail in later lectures.

---

### 10. Ethical Considerations in Machine Learning

Machine Learning is powerful — but with great power comes responsibility.

Bias in data or algorithms can lead to **unfair, discriminatory**, or **unsafe outcomes**.  
Examples include facial recognition systems performing poorly on certain demographics or recommendation algorithms amplifying misinformation.

As ML engineers and researchers, we must:
- Ensure **fairness** and **transparency**.
- Respect **privacy**.
- Evaluate **societal impact**.

Ethical ML is not optional — it’s integral to building trustworthy AI systems.

---

### 11. Summary

In this first week, we explored:
- What Machine Learning is and how it differs from traditional programming.  
- Key types of learning: supervised, unsupervised, semi-supervised, and reinforcement.  
- The ML pipeline and terminology.  
- Concepts of training, testing, underfitting, and overfitting.  
- The importance of ethics in ML.

You are now ready to move on to **Week-02**, where we’ll dive into the mathematics and intuition behind **Linear Regression** — one of the simplest yet most important models in supervised learning.

---

**Suggested Reading:**
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron (Ch. 1)
- Andrew Ng’s *Machine Learning Specialization* (Coursera)
- *Deep Learning* by Goodfellow, Bengio, and Courville (Ch. 1)

---
