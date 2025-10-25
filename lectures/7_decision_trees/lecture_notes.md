# Week 07 – Decision Trees

Decision Trees are among the most intuitive and interpretable machine learning models. They recursively partition the data space into smaller and more homogeneous regions based on feature values. Each partition represents a decision rule, and the final leaves represent the predicted class or value.

This week, we explore how decision trees work for **classification** and **regression**, including the mathematical principles such as **entropy**, **information gain**, **Gini impurity**, and **mean squared error (MSE)** for regression trees.

---

## 1. Intuition Behind Decision Trees

Decision Trees mimic human reasoning when making decisions. Suppose you are trying to decide whether you will enjoy an outdoor activity. You might ask:

- Is it **Sunny**, **Overcast**, or **Rainy**?
- If it is Sunny, is it **too hot**?
- If it is Rainy, is it **too cold**?

Each question divides the possible outcomes into smaller groups. The final decision is made after a series of such splits. This hierarchical, question-based reasoning is exactly how decision trees work.

---

## 2. Example Dataset: Predicting Outdoor Enjoyment

Consider the following dataset for predicting whether a person enjoyed an outdoor activity based on **Weather (Outlook)** and **Temperature**.

| Person | Weather (Outlook) | Temperature | Enjoyed? |
|:-------:|:-----------------:|:------------:|:---------:|
| 1 | Sunny | Hot | No |
| 2 | Sunny | Hot | No |
| 3 | Overcast | Hot | Yes |
| 4 | Rainy | Mild | Yes |
| 5 | Rainy | Cool | Yes |
| 6 | Rainy | Cool | No |
| 7 | Overcast | Cool | Yes |
| 8 | Sunny | Mild | Yes |
| 9 | Sunny | Cool | Yes |
| 10 | Rainy | Mild | Yes |

**Initial distribution:** 7 Yes, 3 No

Our goal is to learn a set of decision rules that can correctly predict “Enjoyed?” for new, unseen data.

---

## 3. Entropy – Measuring Impurity

**Entropy** measures the amount of disorder or impurity in a dataset. It quantifies how mixed the class labels are within a node.

```math
H(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)
```

where:
- $p_{+}$ = proportion of positive examples (Yes)
- $p_{-}$ = proportion of negative examples (No)

Entropy ranges from **0** (perfectly pure) to **1** (completely impure).

### Entropy of the Root Node

We have:

```math

p_+ = \frac{7}{10}, \quad p_- = \frac{3}{10}

```

```math

H(S) = -0.7 \log_2(0.7) - 0.3 \log_2(0.3) \approx 0.881

```

This means the initial dataset still contains uncertainty — it’s not fully pure.

---

## 4. Information Gain – Choosing the Best Split

When a dataset is split on a feature, we expect to reduce uncertainty.  
The **Information Gain (IG)** quantifies this reduction:

\[
IG = H(S) - \sum_{i=1}^{k} \frac{N_i}{N} H(S_i)
\]

where:
- \( H(S) \) is entropy before the split,
- \( H(S_i) \) is entropy of each subset after splitting,
- \( \frac{N_i}{N} \) is the fraction of samples in each subset.

We compute IG for all candidate features and choose the feature that gives the **maximum IG**.

---

### Example: Split by “Outlook”

| Outlook | Yes | No | Total | \(p_+\) | \(p_-\) | Entropy |
|:--------:|:----:|:---:|:------:|:--------:|:--------:|:----------:|
| Sunny | 2 | 3 | 5 | 0.4 | 0.6 | 0.971 |
| Overcast | 2 | 0 | 2 | 1.0 | 0.0 | 0.0 |
| Rainy | 3 | 1 | 4 | 0.75 | 0.25 | 0.811 |

Weighted average entropy after split:

\[
H_{\text{split}} = \frac{5}{10}(0.971) + \frac{2}{10}(0.0) + \frac{4}{10}(0.811) = 0.788
\]
\[
IG = 0.881 - 0.788 = 0.093
\]

This process repeats for other features (like **Temperature**) and the one with higher \(IG\) becomes the root split.

---

## 5. Gini Impurity (Alternative Criterion)

Another popular impurity measure is **Gini impurity**, defined as:

\[
G(S) = 1 - \sum_{i=1}^{k} p_i^2
\]

For our root node:

\[
G(S) = 1 - (0.7^2 + 0.3^2) = 0.42
\]

Gini impurity is computationally faster and is commonly used in libraries such as **scikit-learn**.

---

## 6. Training a Decision Tree

A Decision Tree grows recursively by:
1. Calculating impurity (Entropy or Gini) at the root.
2. Evaluating all possible splits for each feature.
3. Choosing the feature and threshold that maximize information gain.
4. Repeating for each subset until:
   - A node becomes pure, or
   - Maximum depth or minimum sample size is reached.

This recursive partitioning produces a **tree structure** that can make predictions by following decision paths from root to leaf.

---

## 7. Decision Boundaries and Nonlinearity

Decision Trees can model **non-linear relationships**.  
Even if the true boundary is circular or irregular, trees can approximate it by creating multiple axis-aligned splits. This makes them flexible but also prone to overfitting if not regularized.

---

## 8. Cost Function of a Decision Tree

At any point, the objective function for the decision tree can be defined as minimizing the weighted impurity of all nodes:

```math
J = \sum_{m=1}^{M} \frac{N_m}{N} H_m
```

where \(H_m\) can represent entropy or Gini impurity at node \(m\).

---

## 9. Example Final Tree for the Classification Problem


Each branch corresponds to a decision rule derived from maximizing information gain at every step.

---

## 10. Regression Using Decision Trees

While classification trees predict **discrete labels**, **regression trees** predict **continuous numerical values** such as prices, salaries, or temperatures.

The key idea is the same — recursively split the feature space — but the measure of impurity changes from **entropy** to **Mean Squared Error (MSE)**.

---

### Dataset

We use the following small dataset of house prices:

| Size (sqft) | Bedrooms | Price ($1000s) |
|:-------------:|:-----------:|:----------------:|
| 1000 | 2 | 300 |
| 1200 | 2 | 320 |
| 1500 | 3 | 400 |
| 1800 | 3 | 420 |
| 2000 | 3 | 500 |
| 2200 | 4 | 520 |

---

### Step 1: Define Objective Function

A regression tree tries to minimize the **Mean Squared Error (MSE)** within each node:

\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
\]

At each split, the algorithm chooses the threshold that minimizes the **Weighted MSE**:

\[
\text{Weighted MSE} = \frac{N_L}{N} \text{MSE}_L + \frac{N_R}{N} \text{MSE}_R
\]

where \(N_L\) and \(N_R\) are the number of samples in the left and right subsets respectively.

---

### Step 2: Compute MSE for the Entire Dataset

\[
\bar{y}_{\text{root}} = \frac{300 + 320 + 400 + 420 + 500 + 520}{6} = 410
\]

\[
\text{MSE}_{\text{root}} = \frac{(300-410)^2 + (320-410)^2 + (400-410)^2 + (420-410)^2 + (500-410)^2 + (520-410)^2}{6}
\]

\[
= \frac{12100 + 8100 + 100 + 100 + 8100 + 12100}{6} = 6750
\]

---

### Step 3: Evaluate Possible Splits on `Size`

The unique sorted sizes are:
\[
[1000, 1200, 1500, 1800, 2000, 2200]
\]
Possible thresholds are: 1100, 1350, 1650, 1900, 2100.

Below are sample computations for each threshold.

---

#### Split 1: Size < 1100
**Left:** [1000] → [300]  
**Right:** [1200, 1500, 1800, 2000, 2200] → [320, 400, 420, 500, 520]

\[
\bar{y}_L = 300, \quad \text{MSE}_L = 0
\]
\[
\bar{y}_R = 432, \quad \text{MSE}_R = 5216
\]
\[
\text{Weighted MSE} = \frac{1}{6}(0) + \frac{5}{6}(5216) = 4346.67
\]
\[
\Delta_{\text{var}} = 6750 - 4346.67 = 2403.33
\]

---

#### Split 2: Size < 1350
**Left:** [1000, 1200] → [300, 320]  
**Right:** [1500, 1800, 2000, 2200] → [400, 420, 500, 520]

\[
\text{MSE}_L = 100, \quad \text{MSE}_R = 2600
\]
\[
\text{Weighted MSE} = \frac{2}{6}(100) + \frac{4}{6}(2600) = 1766.67
\]
\[
\Delta_{\text{var}} = 6750 - 1766.67 = 4983.33
\]

---

#### Split 3: Size < 1650
**Left:** [1000, 1200, 1500] → [300, 320, 400]  
**Right:** [1800, 2000, 2200] → [420, 500, 520]

\[
\text{MSE}_L = 1866.67, \quad \text{MSE}_R = 1866.67
\]
\[
\text{Weighted MSE} = 1866.67
\]
\[
\Delta_{\text{var}} = 6750 - 1866.67 = 4883.33
\]

---

#### Split 4: Size < 1900
**Left:** [1000, 1200, 1500, 1800] → [300, 320, 400, 420]  
**Right:** [2000, 2200] → [500, 520]

\[
\text{MSE}_L = 2200, \quad \text{MSE}_R = 100
\]
\[
\text{Weighted MSE} = \frac{4}{6}(2200) + \frac{2}{6}(100) = 1500
\]
\[
\Delta_{\text{var}} = 6750 - 1500 = 5250
\]

---

#### Split 5: Size < 2100
**Left:** [1000, 1200, 1500, 1800, 2000] → [300, 320, 400, 420, 500]  
**Right:** [2200] → [520]

\[
\text{MSE}_L = 5168, \quad \text{MSE}_R = 0
\]
\[
\text{Weighted MSE} = \frac{5}{6}(5168) = 4306.67
\]
\[
\Delta_{\text{var}} = 6750 - 4306.67 = 2443.33
\]

---

### Step 4: Best Split and Recursive Partitioning

The split **Size < 1900** yields the **highest variance reduction (5250)** and is therefore chosen as the best split.  

The algorithm then recursively applies the same process on each resulting node until stopping conditions are met — such as reaching a minimum number of samples per leaf or no further reduction in MSE.

---

### Step 5: Predictions

Each **leaf node** in a regression tree predicts the **mean target value** of its samples.  
Thus, prediction for new data involves traversing the tree based on the feature thresholds until a leaf node is reached.

---

## 11. Summary

- Decision Trees classify data by splitting the feature space based on impurity reduction.
- For classification, impurity is measured using **Entropy** or **Gini**.
- For regression, impurity is measured using **Mean Squared Error (MSE)**.
- The model grows recursively by choosing the split that best reduces impurity.
- While intuitive and interpretable, Decision Trees can **overfit**; pruning or ensemble methods like **Random Forests** are often used to improve generalization.

---

## References

- A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3rd Edition, O’Reilly Media, 2023.
- Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)


