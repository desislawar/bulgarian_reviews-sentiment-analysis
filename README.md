# bulgarian_reviews-sentiment-analysis
Sentiment analysis on movies' reviews from Bulgaria

This project presents a binary sentiment classifier built on movie reviews and compares the traditional baseline models to a fine-tuned multilingual transformers.

# Overview

The baseline is implemented as a standard ML pipeline combining TF-IDF vectorization and a linear classifier(Logistic Regression/ SVM), while the choice for fine-tuned multilingual transformer is mDeBERTa-v3. 

The primary objective is to classify user-generated reviews as positive or negative, addressing inherent challenges such as ambiguous language, class imbalance and noisy annotation, and systematically compare the performance between interpretable classical ML models and black-box deep learning models.

# Dataset
The experiments use publicly available dataset, containing bulgarian-language user-generated reviews. The dataset is hosted on Hugging Face(https://huggingface.co/datasets/DGurgurov/bulgarian_sa). 
The dataset comes into three folds:

|Split          |        Samples|
|-------------- |-------------- |
| Training      |          5,412|
| Validation    |            838|
| Test          |          1,673|

An initial examination of the training subset revealed a substantial class imbalance in favor of the positive reviews as follows:

|Review         |        Samples|
|-------------- |-------------- |
| Positive      |          4,587|
| Negative      |            825|

This imbalace motivated the use of class-weighted training and macro averaged F1-score as trustworthy metrics for the purpose of this project. 

# Methodology

Each end-to-end model workflow is serialized as a unified pipeline object.

## Baseline models
The preprocessing for the baseline projects is identical. It consists of the following actions - lowercasing, whitespace trimming and reduction. The **ngram_range**, **n-grams lenght**, **regularization parameter** and **custom class weight** are tuned via **GridSearch** and out-of-fold predicted probabilities are generated for predicted probability/decision score in order to maximize macro averaged f-1 score. Here are the objectives for both baseline models: 

|Hyperparameters          |LR             |SVM            |
|------------------------ |-------------- |-------------- |
| **ngram_range**         |(2,7)          |(1,7)          |
| **C**                   |3.0            |0.3            | 
| **class weight{0, 1}**  |0: 3.0, 1: 1.0 |0: 3.0, 1: 1.0}|
| **decision values**     |0.53999        | 0.01001       |

## Transformer Fine-Tuning
The approach differs as here a fine-tuning on a pretrained multilingual transofrmer is performed. 
Unlike TF-IDF, transformer representations incorporate contextual information, which alllows the model to represent words differently depending on their surrounding text. 
Fine-tuning was performed using Pytorch. Here a different set of hyperparameters is optimized - **learning rate**, **number of training epochs**, **weight_decay** again with the objective to maximize average f1_macro. Under this condition, the following values were selected for the hyperparameters:

|Hyperparameters          |value          |
|------------------------ |-------------- |
| **learning rate**       |2e-05          |
| **num_train_epochs**    |3.0            |
| **weight decay**        |0.01           |

# Experimental Results 

## Baseline 

|Metrics                  |LR             |SVM            |
|------------------------ |-------------- |-------------- |
| accuracy                |0.9331         |0.9366         |
| macro avg f1-score      |0.8966         |0.9002         | 

| confusion matrix        |[[ 284   55]   |[[ 278   61]
                           [  57 1277]]   |[  45 1289]]   |
