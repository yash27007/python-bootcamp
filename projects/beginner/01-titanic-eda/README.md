# 01 – Titanic Survival EDA

**Difficulty:** Beginner  
**Domain:** Exploratory Data Analysis  
**Dataset:** [Kaggle Titanic](https://www.kaggle.com/c/titanic)

## Objective

Explore the famous Titanic dataset to understand which factors influenced passenger survival. Practice real-world data cleaning, visualisation, and insight extraction — skills every data scientist uses daily.

## What You'll Learn

- Loading and inspecting a dataset with Pandas
- Handling missing values
- Univariate and bivariate analysis
- Visualising distributions with Matplotlib and Seaborn
- Forming and testing hypotheses from data

## Approach

1. Load data and inspect shape, dtypes, and missing values
2. Impute/drop missing values (Age, Cabin, Embarked)
3. Analyse survival rates by: Sex, Pclass, Age, Fare, Embarked, SibSp, Parch
4. Correlation heatmap for numeric features
5. Summarise key findings

## Key Questions to Answer

- What was the overall survival rate?
- Did gender affect survival? (Hint: "women and children first")
- Did passenger class (1st, 2nd, 3rd) affect survival?
- Were younger passengers more likely to survive?
- Did family size matter?

## Prerequisites

Section 01 (Python Basics), Section 03 (Data Analysis — Pandas, Seaborn)

## How to Run

```bash
# From repo root
uv sync
jupyter lab projects/beginner/01-titanic-eda/notebook.ipynb
```

## Data Download

```bash
# Option 1: Kaggle CLI
kaggle competitions download -c titanic -p projects/beginner/01-titanic-eda/data/

# Option 2: Manual
# Download from https://www.kaggle.com/c/titanic/data
# Place train.csv and test.csv in projects/beginner/01-titanic-eda/data/
```

## Expected Results

| Metric | Value |
|--------|-------|
| Overall survival rate | ~38% |
| Female survival rate | ~74% |
| Male survival rate | ~19% |
| 1st class survival | ~63% |
| 3rd class survival | ~24% |
