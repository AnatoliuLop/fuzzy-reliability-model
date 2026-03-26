# Weighted Fuzzy Reliability Model for Prediction Analysis

## Overview

This project presents an implementation of an **enhanced weighted fuzzy model** for evaluating the **reliability of predictions in decision-making systems**.

The work is based on a bachelor's thesis focused on improving classification models by introducing mechanisms for:
- handling uncertainty,
- estimating prediction reliability,
- and dynamically adapting to data.

The proposed model extends the original approach by Maimon et al. with additional techniques aimed at improving **accuracy, robustness, and interpretability**.

---

## Motivation

In real-world systems, data is often:
- incomplete,
- noisy,
- or uncertain.

Traditional models assume binary correctness (true/false), which is insufficient in many scenarios.

This project introduces a **fuzzy reliability framework**, where each prediction is associated with a **confidence score**, allowing better decision-making under uncertainty.
---

## Core Idea

The model combines:

- **Fuzzy logic** → to handle uncertainty  
- **Probability theory** → to estimate class likelihoods  
- **Information theory (Mutual Information)** → to detect feature dependencies  

Instead of relying on a single attribute, the model aggregates **weighted contributions of all features**, making it more robust and informative.

---

## Key Contributions

### 1. Dynamic Feature Dependency Detection
- Automatically selects dependent attribute pairs  
- Uses **Mutual Information (MI)** to identify informative relationships  
- Improves model expressiveness and accuracy :contentReference
---

### 2. Probability Interpolation Mechanism
The model introduces interpolated probabilities:

- Prior probabilities  
- Conditional probabilities  
- Pairwise probabilities  

These are dynamically updated using weighted parameters:
- α (priors)
- β (conditionals)
- γ (pair interactions)

This allows the model to:
- remain stable with small data
- adapt as new data arrives :contentReference[oaicite:3]{index=3}  

---

### 3. Reliability Evaluation

Each prediction is assigned a **fuzzy reliability score**, representing confidence in the decision.

The model supports multiple evaluation modes:

- **Baseline** – static evaluation  
- **Retrained** – incremental learning  
- **Batched-end** – learning from latest data only  

This enables analysis of how reliability changes under different learning strategies. :contentReference

---

### 4. Hyperparameter Optimization

- Automatic tuning using **Grid Search**
- Validated via **Cross-Validation**
- Improves generalization and performance

---

## Experimental Evaluation

The model was tested on public datasets from the **UCI Machine Learning Repository**:

- **Car Evaluation dataset**
- **Balance Scale dataset**

Evaluation metrics include:
- Classification accuracy
- Average reliability score
- Comparison across different training strategies

---

## Results

Experimental results show that the proposed model:

- Achieves **higher accuracy** than the original approach  
- Better identifies **uncertain or low-confidence predictions**  
- Improves robustness in noisy and limited-data scenarios  

This confirms the effectiveness of combining:
- fuzzy logic,
- probabilistic modeling,
- and feature interaction analysis.

---

## System Architecture

The system consists of several core components:

- **FuzzyReliabilityModel** – main classification and reliability engine  
- **ReliabilityUtils** – evaluation logic  
- **OutputCSV** – result analysis and reporting  
- **Config** – dataset configuration  

The entire solution is implemented in **C++**.

---

## Use Cases

- Decision support systems  
- Risk analysis  
- Data quality evaluation  
- Systems working with uncertain or incomplete data  

---

## Academic Context

This project was developed as part of a Bachelor's thesis at:

**University of Žilina (UNIZA)**  
Faculty of Management Science and Informatics  

The work contributes to research in:
- prediction reliability,
- fuzzy systems,
- and uncertainty-aware machine learning.

---

## Author
Anatolii Lopatiuk
