# Topic - Quick Reference
*Generated: 2025-11-25 18:24:48*

## Curriculum Structure

### **Curriculum Overview**
*   **Total Duration:** 2 Semesters (approx. 30 weeks)
*   **Prerequisites:** High School Mathematics (Pre-calculus, basic statistics) and basic computer literacy.
*   **Target Outcome:** A student who can build basic ML models from scratch, understands the math behind them, and uses Python proficiently.
---
## **Semester 1: The Toolbox (Foundations)**
*Goal: Master the languages of AI (Python & Mathematics).*
### **## Module 1: Programming for Data Science**
*This is not just generic coding; it is coding with a focus on data manipulation and vectorization.*
*   **### Core Concepts**
    *   Python Syntax & Data Structures (Lists, Dictionaries, Sets). [Level 1]
    *   **Vectorization:** Introduction to NumPy and matrix operations. [Level 3]
    *   Data Manipulation: Using Pandas for loading and cleaning datasets. [Level 2]
    *   Visualization: Matplotlib/Seaborn to view data distributions. [Level 1]
*   **### Learning Objectives**
    *   Write efficient Python scripts without relying on loops for mathematical operations.
    *   Clean a dataset (handling missing values, format inconsistencies).
*   **### Assessment**
    *   **Project:** Analyze a real-world dataset and produce a visual report.
### **## Module 2: Mathematics for Machine Learning I (Linear Algebra)**
*ML models represent matrices interacting. This module explains that process.*
*   **### Core Concepts**
    *   Vectors, Matrices, and Tensors. [Level 1]
    *   Matrix Multiplication (Dot product vs. Cross product). [Level 3]
    *   Eigenvalues and Eigenvectors. [Level 2]
    *   Dimensionality Reduction intuition (PCA basics). [Level 1]
*   **### Learning Objectives**
    *   Perform matrix operations by hand and in NumPy.
    *   Understand geometric interpretations of vectors (projections, spaces).
### **## Module 3: Mathematics for Machine Learning II (Calculus & Probability)**
*Understanding how models learn (Calculus) and how they handle uncertainty (Probability).*
*   **### Core Concepts**
    *   **Derivatives & Gradients:** Understanding slope as rate of change for optimization. [Level 3]
    *   Chain Rule: How changes propagate through a system. [Level 2]
    *   Bayes Theorem: Updating beliefs with new data. [Level 2]
    *   Distributions: Normal, Bernoulli, and Poisson distributions. [Level 1]
*   **### Learning Objectives**
    *   Calculate gradients for simple functions.
    *   Calculate conditional probabilities for real-world scenarios.
---
## **Semester 2: Introduction to Intelligence**
*Goal: Apply the foundations to build the first actual AI/ML systems.*
### **## Module 4: Classical AI & Problem Solving**
*Understanding symbolic AI and state machines.*
*   **### Core Concepts**
    *   Intelligent Agents & Environments (PEAS model). [Level 1]
    *   **Search Algorithms:** BFS, DFS, A* Search (Pathfinding). [Level 3]
    *   Game Theory: Minimax Algorithm. [Level 2]
    *   Constraint Satisfaction Problems. [Level 1]
*   **### Learning Objectives**
    *   Implement an A* search to solve a maze.
    *   Build an agent that plays a simple board game effectively.
### **## Module 5: Foundations of Machine Learning (Supervised)**
*The core of modern AI application.*
*   **### Core Concepts**
    *   **Regression:** Linear Regression, Cost Functions (MSE), and Gradient Descent. [Level 3]
    *   **Classification:** Logistic Regression, K-Nearest Neighbors (KNN). [Level 2]
    *   Model Evaluation: Train/Test Split, Accuracy vs. Precision/Recall. [Level 3]
    *   Overfitting vs. Underfitting (Bias-Variance Tradeoff). [Level 2]
*   **### Learning Objectives**
    *   Build a Linear Regression model from scratch using only NumPy to understand the math.
    *   Use Scikit-Learn to train a classifier on standard datasets.
### **## Module 6: Unsupervised Learning & Clustering**
*Finding patterns in data without labels.*
*   **### Core Concepts**
    *   Clustering: K-Means Algorithm. [Level 2]
    *   Dimensionality Reduction: Principal Component Analysis (PCA) in practice. [Level 2]
    *   Anomaly Detection basics. [Level 1]
*   **### Learning Objectives**
    *   Group customers into segments based on purchasing data without knowing categories beforehand.
### **## Module 7: AI Ethics & The Modern Landscape**
*Ethical engineering and architectural overview.*
*   **### Core Concepts**
    *   Bias and Fairness: How data bias creates model bias. [Level 2]
    *   Explainability (XAI): Black box vs. White box models. [Level 1]
    *   Overview of Deep Learning & LLMs: High-level understanding of Neural Nets and Transformers. [Level 1]
*   **### Learning Objectives**
    *   Critique a hypothetical AI system for potential ethical failures.
    *   Use a pre-trained Large Language Model (via API) for a structured task.
---
## **Depth Markers Key**
*   [Level 1] **Basic Awareness:** Understand the definition and use case.
*   [Level 2] **Working Knowledge:** Can implement with libraries/tools.
*   [Level 3] **Deep Mastery:** Can derive the math or build from scratch without libraries.
## **Cross-Reference & Progression**
*   **Module 1 (Python)** is used in subsequent modules.
*   **Module 2 (Linear Algebra)** is the direct prerequisite for **Module 5 & 6** (Data representation).
*   **Module 3 (Calculus)** is strictly required for understanding Gradient Descent in **Module 5**.
*   **Module 4 (Search)** teaches algorithmic thinking needed for optimization.
## **Capstone Assessment Suggestions**
    *   **Data:** Raw housing data (CSV).
    *   **Task:** Clean data (Module 1), visualize correlations (Module 1), select features (Module 2), train a Regression model (Module 5), and evaluate error (Module 3/5).
    *   **Data:** Text dataset of emails.
    *   **Task:** Convert text to vectors (Module 2), use Naive Bayes or Logistic Regression (Module 3/5) to classify as Spam/Not Spam.

## Key Resources

## Study Tips

- Follow the curriculum structure sequentially
- Cross-reference multiple sources for each concept
- Practice with hands-on examples
- Review and revise regularly
