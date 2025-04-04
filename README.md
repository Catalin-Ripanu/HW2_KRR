# Knowledge Representation and Reasoning: Homework Assignment 2

## Overview

This homework is divided into two main parts:
1. Implementation and application of the Expectation-Maximization (EM) algorithm for handling missing data
2. Critical analysis of various embedding techniques used in machine learning

## Part I: Expectation-Maximization Algorithm (7 points)

### Problem 1
Theoretical explanation of how the EM algorithm can be used to estimate parameters with missing data.

**Tasks:**
- Explain EM algorithm application for estimating mean and variance of Y and correlation between X and Y when Y has missing values
- Justify why log-likelihood increases after each iteration

### Problem 2
Implementation of the EM algorithm for imputing missing values in synthetic data.

**Requirements:**
- Create a synthetic dataset and remove some values to simulate missing data
- Implement the EM algorithm with:
  - Random initialization of missing values
  - E-step (estimating missing values)
  - M-step (recomputing parameters)
  - Output of imputed dataset and final parameter estimates

**Evaluation:**
- Visualize original data with missing values vs. imputed dataset
- Compare imputed values with true values
- Plot log-likelihood values to demonstrate convergence

### Problem 3
Application of the EM algorithm to a real-world dataset with missing values.

**Requirements:**
- Choose a real-world dataset with missing values (suggestions provided)
- Preprocess data as necessary
- Apply your EM algorithm implementation
- Visualize dataset before and after imputation
- Analyze imputed values against domain knowledge
- Evaluate imputation impact on dataset structure

## Part II: Analysis of Embedding Techniques (3 points)

Write a critical essay (1,000-1,500 words) analyzing different embedding techniques across various data types.

**Essay Structure:**
1. **Introduction to Embeddings**
   - General concept of embeddings and dimensionality reduction
   - Examples across different data types

2. **Analysis of Different Types of Embeddings**
   - Compare at least three embedding techniques
   - For each technique, cover:
     - Working mechanism
     - Applicable data types
     - Real-world applications
     - Advantages and limitations

3. **Real-world Performance Comparison**
   - Provide a scenario comparing at least two embedding techniques

4. **Conclusion**
   - Effect of embedding technique on downstream ML tasks
   - Universality vs. domain-specificity of embeddings

**Requirements:**
- Include at least 3 references
- Submit as PDF document

## Submission Requirements

You must submit:
1. PDF file containing answers to Part I (1-3) with plots and visualizations
2. Program file with your EM algorithm implementation
3. PDF file containing the embedding techniques essay

## Important Notes

- You must explicitly indicate if you used any generative models in your answers or code
- If generative models were used, clearly explain your own contribution
- Be prepared to defend your homework during the last laboratory session
- You will be questioned on both the written part and the code
