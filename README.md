# Feature-Importance-Analysis
Machine Learning Pipeline Optimization with mlr3 for feature Importance analysis

This repository contains R scripts for feature selection and model tuning using the mlr3 ecosystem. The project demonstrates two approaches to feature selection (filter and wrapper methods) and compares their performance on classification tasks.

## Features
Feature importance analysis using decision trees and random forests

Graph-based pipeline construction with mlr3pipelines

Hyperparameter tuning with bbotk

Benchmarking of different feature selection methods

Visualization of results using autoplot

## Scripts
The repository contains code for:

Feature Importance Analysis:

Using decision trees (classif.rpart)

Using random forests (classif.ranger)

## Graph-Based Pipeline Construction:

Building a GraphLearner that combines feature selection with model training

Tuning the number of features to select

## Wrapper Methods:

Sequential feature selection with stagnation termination

Genetic algorithm-based feature selection

Random search-based feature selection

## Benchmarking:

Comparing filter vs. wrapper approaches

Comparing genetic search vs. random search

Baseline comparisons with simple models

### The scripts generate visualizations of:

Feature importance scores

Performance vs. number of features selected

Benchmark comparisons between different approaches

## Datasets
The code demonstrates the techniques on:

spam dataset (from mlr3)

sonar dataset (from mlr3)

## License
MIT