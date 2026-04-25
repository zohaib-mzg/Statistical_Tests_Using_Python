# 📊 Statistical Tests Using Python

> A comprehensive collection of hypothesis testing and model selection implementations in Python for Advanced Statistics — covering parametric tests, non-parametric tests, and automated variable selection techniques.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Pandas%20%7C%20SciPy%20%7C%20Scikit--learn-orange)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Scripts & Descriptions](#scripts--descriptions)
  - [Backward Elimination](#backward-elimination)
- [How It Works — Algorithm Walkthrough](#how-it-works--algorithm-walkthrough)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Statistical Concepts Reference](#statistical-concepts-reference)
- [License](#license)

---

## Overview

This repository provides clean, well-structured Python implementations of statistical hypothesis tests used in **Advanced Statistics** and **Quantitative Research**. The goal is to make it easy for students, data scientists, and researchers to understand, run, and extend classical statistical methods — without relying on black-box libraries.

Whether you need to build a **parsimonious regression model** using backward elimination or apply formal **F-tests for variable significance**, this project gives you readable, educational code with transparent logic.

---

## Features

- ✅ Automated **Backward Elimination** for multiple linear regression
- ✅ **F-test based hypothesis testing** for variable significance
- ✅ Computes **Full Model** and **Reduced Model** metrics (SSR, SSE, S)
- ✅ Iterative variable removal with **F-critical threshold** at α = 0.05
- ✅ Prints **model coefficients** at each stage for transparency
- ✅ Works with any CSV dataset (headerless format)
- ✅ Built on standard scientific Python stack (NumPy, Pandas, SciPy, Scikit-learn)

---

## Repository Structure

```
Statistical_Tests_Using_Python/
│
├── Backward_Elimination.py   # Automated backward variable selection via F-tests
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

---

## Scripts & Descriptions

### Backward Elimination

**File:** `Backward_Elimination.py`

Implements the **Backward Elimination** method for multiple linear regression model selection. Starting with all predictors included, the algorithm iteratively removes the least statistically significant variable — the one whose removal causes the smallest reduction in SSR — until all remaining variables pass the F-test at the 5% significance level.

#### Functions

| Function | Description |
|---|---|
| `linearRegModel(df)` | Fits a linear regression model and returns intercept, coefficients, and predictions |
| `calcFullModel(df)` | Computes SSR, SSE, and residual variance (S) for the full model |
| `calcReducedModel(cols, df, SSR_full)` | Computes conditional SSR for each candidate variable to remove |
| `HypothesisTest(df)` | Performs the F-test and decides whether to remove the selected variable |
| `showOutput(df)` | Prints the model coefficients (b0, b1, b2, ...) |
| `processRun()` | Main driver function — reads the CSV, runs elimination stages, prints results |

---

## How It Works — Algorithm Walkthrough

The backward elimination procedure follows these steps:

```
1. Start with the FULL MODEL (all predictors included)
2. For each predictor xᵢ:
      a. Fit the REDUCED MODEL without xᵢ
      b. Compute SSR(xᵢ | others) = SSR_full − SSR_reduced
3. Select the predictor with the MINIMUM conditional SSR (least useful variable)
4. Compute the F-statistic:
         F = SSR(xᵢ | others) / S_full
5. Compare F_calc to F_crit(α=0.05, df1=1, df2=n−k−1):
      - If F_calc ≤ F_crit → REMOVE the variable (not significant)
      - If F_calc > F_crit → KEEP all remaining variables (all are significant)
6. Repeat from step 2 with the reduced set of predictors
7. Stop when only one variable remains OR all remaining variables are significant
```

---

## Prerequisites

Make sure you have **Python 3.7+** installed, along with the following libraries:

```
numpy
pandas
scipy
scikit-learn
```

---

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/zohaib-mzg/Statistical_Tests_Using_Python.git
cd Statistical_Tests_Using_Python
```

**2. Install dependencies:**

```bash
pip install numpy pandas scipy scikit-learn
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the backward elimination script from the terminal:

```bash
python Backward_Elimination.py
```

When prompted, enter the path to your CSV data file:

```
Enter the file name: your_data.csv
```

The script will iterate through elimination stages, printing F-statistics, decisions, and final model coefficients at each step.

---

## Input Data Format

The script expects a **headerless CSV file** where:

- **Column 0** is the **dependent variable (Y)**
- **Columns 1 onward** are the **independent variables (X₁, X₂, ..., Xₖ)**

Example (`data.csv`):

```
25.3, 1.2, 3.4, 7.8
30.1, 2.1, 4.5, 8.0
22.7, 0.9, 3.1, 6.5
...
```

> **Note:** Ensure your data has no missing values before running the script.

---


---

## Statistical Concepts Reference

| Term | Definition |
|---|---|
| **SSR** | Sum of Squares due to Regression — explained variance |
| **SSE** | Sum of Squares due to Error — unexplained variance |
| **S (MSE)** | Mean Squared Error = SSE / (n − k − 1) |
| **F-statistic** | Ratio of explained to unexplained variance for a single predictor |
| **F-critical** | Threshold value from F-distribution at α = 0.05 |
| **Backward Elimination** | Model selection starting with all variables, removing one at a time |
| **Conditional SSR** | Contribution of a variable given all others are in the model |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
---

<p align="center">Made with 🧮 for students and researchers in Advanced Statistics</p>
