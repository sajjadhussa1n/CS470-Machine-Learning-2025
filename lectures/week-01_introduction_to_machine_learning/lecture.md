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

![](./images/traditional_approach.png)

*Figure 1: In traditional programming, data and manually defined rules are combined to produce decisions or outputs.*

![](./images/ml_approach.png)

*Figure 2: In machine learning, data and known decisions (labels) are provided to a learning algorithm to produce a model that can generalize to new data.*

**Source:** Adapted from *Aurélien Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow," 3rd Edition, O’Reilly, 2023.*

---
### When to Use Machine Learning

Machine learning excels when:
- Rules are difficult to define or maintain manually (e.g., **spam filtering**).
- The problem is inherently complex (e.g., **speech recognition**).
- The system must **adapt to changing environments** or new data.
- We want to **extract insights from large datasets**, beyond human analytical capacity.

If a task involves pattern recognition, probabilistic decision-making, or adapting to new data, ML is often the right choice.

---


### Types of Machine Learning

Machine Learning can be broadly categorized based on the type of feedback available from the data.

#### Supervised Learning

In **supervised learning**, we have both input features (**X**) and target outputs (**Y**).  
The model learns a mapping function:  

**f: X --> Y**

and is evaluated by how accurately it predicts \( Y \) for unseen \( X \).

![](./images/supervised.png)

*Figure 3: Supervised Learning.*

**Source:** Adapted from *Aurélien Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow," 3rd Edition, O’Reilly, 2023.*

**Examples:**
- Predicting house prices (Regression)
- Email spam detection (Classification)

**Algorithms:** Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines, Neural Networks.

##### Classification vs Regression

Supervised learning can be divided into **classification** and **regression** problems.

| Classification | Regression |
|----------------|-------------|
| Predicts **discrete classes** (e.g., spam/not spam) | Predicts **continuous numeric values** (e.g., house prices) |
| Output is a **label** | Output is a **real number** |
| Example tasks: Spam detection, Image classification, Disease diagnosis | Example tasks: Price prediction, Temperature forecasting, Stock value estimation |
| Evaluation metrics: Accuracy, Precision, Recall, F1-score | Evaluation metrics: MSE, RMSE, MAE, R² |

Both are forms of **supervised learning** since they rely on labeled training data.

#### Unsupervised Learning

Here, only the inputs \( X \) are available — no labeled outputs.  
The goal is to **discover hidden structures or patterns** within the data.

![](.images/unsupervised.png)

**Examples:**
- Grouping customers by purchase behavior (Clustering)
- Reducing dimensionality for visualization (PCA)

**Algorithms:** K-Means, DBSCAN, PCA, Autoencoders.

#### Semi-Supervised Learning

A combination of labeled and unlabeled data. Often used when labeling is expensive but large amounts of unlabeled data are available.

![](.images/semisupervised.png)

**Example:** Training a face recognition model using a small set of labeled faces and many unlabeled ones.

#### Reinforcement Learning

In this paradigm, an **agent** learns by interacting with an environment.  
It takes actions, receives rewards or penalties, and gradually learns a policy that maximizes cumulative rewards.

![](.images/reinforcement.png)

Key terms:
- **Agent:** The learner or decision-maker.
- **Environment:** Everything the agent interacts with.
- **Policy:** Strategy used to determine the next action.
- **Reward:** Feedback signal for success or failure.

**Examples:**
- Game playing (AlphaGo)
- Robotics control
- Traffic light optimization

---

### Key Terminology

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

## Example Datasets

- **Supervised:** Housing prices, sentiment analysis, image classification  
- **Unsupervised:** Customer segmentation, topic modeling in news articles  
- **Reinforcement:** Game simulations, robotic navigation  

---

## Challenges in Machine Learning

Building a successful ML system involves more than just choosing an algorithm. Data quality, representativeness, and model complexity all play critical roles.

---

### Insufficient Quantity of Training Data

ML algorithms require large, diverse datasets to generalize well.  
Simple tasks may require thousands of examples, while complex ones (like speech recognition) may need millions.

![](.images/dataset_size.png)

A Microsoft study demonstrated that even simple models can perform well on difficult tasks when trained with large amounts of high-quality data — reinforcing the idea that **data often matters more than algorithms**.

---

### Non-Representative Training Data

If the training data doesn’t reflect the real-world cases we aim to predict, the model will fail to generalize.

- **Sampling Noise:** Random variability due to small datasets.  
- **Sampling Bias:** Systematic errors introduced by flawed sampling methods.

Example: In the 1936 U.S. election, *Literary Digest* incorrectly predicted a Republican win because their survey disproportionately sampled wealthy individuals. The actual winner, Franklin D. Roosevelt (Democrat), won by a landslide.

---

### Poor-Quality Data

Noisy, inconsistent, or incomplete data can degrade model performance.

Typical remedies include:
- Removing or correcting outliers.  
- Handling missing values through imputation (e.g., mean or median).  
- Dropping features or instances that lack reliable information.

Data cleaning and preprocessing often consume a large portion of an ML practitioner’s time.

---

### Irrelevant Features

The quality of a model’s input features largely determines its success.  
“**Garbage in, garbage out**” aptly summarizes this.

Good **feature engineering** involves:
- **Feature selection:** Identifying the most useful existing features.  
- **Feature extraction:** Transforming or combining features (e.g., PCA).  
- **Feature creation:** Deriving new meaningful variables.

---

### Overfitting the Training Data

**Overfitting** occurs when the model learns the training data too well—including its noise and anomalies—resulting in poor performance on unseen data.

![](.images/over_fitting.png)

Overfitting usually arises from:
- Models that are too complex.
- Small or noisy datasets.

---

### Preventing Overfitting

To reduce overfitting:
- **Simplify the model:** Use fewer parameters or apply regularization.  
- **Improve data quality:** Gather more diverse samples and remove noise.  
- **Apply regularization:** Penalize overly complex models to keep weights small.

Regularization introduces a **hyperparameter** that controls the strength of the penalty, balancing model flexibility and generalization.

---

### Underfitting the Training Data

**Underfitting** happens when the model is too simple to capture the underlying data structure.  
This leads to poor performance even on the training set.

To fix underfitting:
- Use more expressive models (e.g., add polynomial terms).  
- Provide richer features.  
- Reduce constraints (less regularization).

---

### Testing and Validating Models

To assess how well a model generalizes to new data, we divide the available data into:
- **Training set:** Used for learning parameters.  
- **Test set:** Used for final evaluation.

If the model performs well on training but poorly on test data → **overfitting**.  
Typical split: 80% training, 20% testing (though it depends on dataset size).

---

### Hyperparameter Tuning and Model Selection

Using the test set to repeatedly adjust model parameters can lead to **test set overfitting**.  
To avoid this, we use:

- **Holdout validation:** Split into training, validation, and test sets.  
- **Cross-validation:** Train and evaluate across multiple folds of the dataset to obtain a more reliable estimate of model performance.

These methods help identify the best model configuration before final testing.

---

### No Free Lunch Theorem

The **No Free Lunch (NFL)** theorem, proposed by David Wolpert (1996), states that:

> No single learning algorithm performs best for every problem.

Each model makes assumptions about the data. For example, linear models assume linear relationships, while neural networks can represent more complex, nonlinear structures.  
Thus, the best algorithm depends entirely on the specific dataset and problem context.

In practice:
- Make reasonable assumptions about your data.
- Experiment with multiple models.
- Use validation to guide model choice.

---

## Summary of Week 1

This week, we learned:
- The foundations and definitions of machine learning.
- The distinctions among supervised, unsupervised, semi-supervised, and reinforcement learning.
- Key challenges including data quality, overfitting, and underfitting.
- Model evaluation through data splitting and validation.
- The importance of assumptions in model selection, as stated by the No Free Lunch theorem.

**Takeaway:**  
> Successful machine learning is built on high-quality data, thoughtfully designed models, and rigorous evaluation practices.

---

**Suggested Reading:**
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron (Ch. 1)
- Andrew Ng’s *Machine Learning Specialization* (Coursera)
- *Deep Learning* by Goodfellow, Bengio, and Courville (Ch. 1)

---
