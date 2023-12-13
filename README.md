![Python](https://img.shields.io/badge/python-v3.7+-brightgreen.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.0+-blue.svg)
![Keras](https://img.shields.io/badge/keras-v2.3+-purple.svg)
![Sklearn](https://img.shields.io/badge/scikit_learn-v0.22+-lightgrey.svg)
![NLTK](https://img.shields.io/badge/NLTK-v3.4+-blueviolet.svg)
![Pandas](https://img.shields.io/badge/pandas-v1.0+-yellow.svg)
![NumPy](https://img.shields.io/badge/numpy-v1.18+-orange.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-v3.2+-red.svg)


# 🧠 Neural Network Analysis for Predictive Modeling 

The project aims to predict the popularity of a movie based on it's overview text. It involves a thorough analysis of a movie dataset, exploring various aspects of data preprocessing, model building, training, and evaluation.

## 📉 Results (don't judge a book by its cover)

This project demonstrates that **it is very hard to predict the popularity of a movie based on its overview text**.

## Approaches
   - **Regression** => Target: average_vote
   - **Classification** => Target: poplarity_class

## Contents

1. 📚 **Preparation**
   - Loading the Movies dataset from a GitHub repository.
   - Initial exploration of dataset characteristics, including data types, correlations, and handling missing values.
   - Transforming the data by dropping unnecessary columns, creating new features, and visualizing distributions.

2. 🔄 **Data Preprocessing**
   - Feature engineering by converting text data to numerical representations (text vectorization) for the movie overviews and titles.
   - Normalization of numeric features for consistent scaling.
   - Splitting the dataset into training and test sets.

3. 🏗️ **Model Building and Training**
   - Construction of various neural network models for both regression (predicting `vote_average`) and classification (predicting `popularity_class_label`).
   - Models include simple dense networks, LSTM, embedding layers, and convolutional networks, showcasing a range of deep learning techniques.
   - Training the models using different features sets: numeric features, overview text, and title text.

4. 📊 **Evaluation and Results Analysis**
   - Evaluation of models using metrics such as R-squared, explained variance for regression, and accuracy, confusion matrix, and classification report for classification.
   - Visualization of training and validation accuracy over epochs.
   - Comparative analysis of different models based on their performance on the test dataset.

## 🌟 Highlights of the Project

- **Comprehensive Data Analysis**: Detailed examination and transformation of a complex movie dataset.
- **Diverse Model Architectures**: Exploration of various neural network structures tailored for specific types of features.
- **In-depth Model Evaluation**: Extensive analysis of model performance, providing insights into their effectiveness in predictive modeling.

## ✨ Conclusion and Future Work 

The analysis provides valuable insights into the performance of different types of neural network architectures for both regression and classification tasks. Future work could explore further refinement of models, incorporation of additional features, and application to other datasets or predictive scenarios.
