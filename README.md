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

|Hyperparameters          |LR             |SVM               |
|------------------------ |---------------- |--------------- |
| **ngram_range**         |(2,7)            |(1,7)           |
| **C**                   |3.0              |0.3             | 
| **class weight{0, 1}**  |{0: 3.0, 1: 1.0} |{0: 3.0, 1: 1.0}|
| **decision values**     |0.53999          | 0.01001        |

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
| Accuracy                |0.9331         |0.9366         |
| Macro-F1                |0.8966         |0.9002         | 
| Macro Precision         |0.8958         |0.9077         |
| Macro Recall            |0.8975         |0.8932         |

Class specific results: 

|Metric / Class           |LR     |       |SVM    |       |
--------------------------|-------|-------|-------|-------|
| Class                   |0      |1      |0      |1      |
| Precision               |0.8328 |0.9573 |0.8607 |0.9548 |
| Recall                  |0.8378 |0.9573 |0.8201 |0.9663 | 
| F1                      |0.8353 |0.9580 |0.8399 |0.9605 |

Confusion matrices for both baselines:

1. LR
<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/98003a50-d851-45fe-86f7-6cf9981b400e" />


2. SVM
<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/a3254a6f-e7ec-489c-bcdc-4c9dd7f23d17" />

The overall accuracy for both baseline models exceeded 93% but the breakdown by class revealed the stark performance disparity between the two classes. This underlines the importance of class-specific metrics and macro-f1 when handling class imbalance data. Of course, there are specific techniques that could tackle such obstacles but given the colorful language used by users in movie reviews, a cautious and well-thought through approach is advised when applying methods to address class imbalance problems. :) 

## Transformer Results 
|Metrics                  |mDeBERTa       |
|------------------------ |-------------- |
| Accuracy                |0.9588         |
| Macro-F1                |0.9342         |
| Macro Precision         |0.9487         |
| Macro Recall            |0.9213         |

Class specific results: 

|Metric / Class           |   0   |   1   |
--------------------------|-------|-------|
| Precision               |0.9327 |0.9647 |
| Recall                  |0.8584 |0.9843 |
| F1                      |0.8940 |0.9744 |

Confusion matrix:

<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/cdee090d-a1ed-4353-90bd-78d691b29903" />

# Error Analysis

The manual inspection of false-positives and false-negatives from the baseline models shed light to the limitations of such feature extraction method - the context blindness. Because these algorithms evalueate words independently, they lack understanding how surrounding words shape the general attitude within the reviews.
Since the errors from the transformer moder were only 69, this allowed manual review for each error. The nature of the error was subjectively categorized. According to the results the predominant categories were mixed sentiment and insufficient context accounting for 53.6% of the errors, followed by model limitation(23.2%) and potentially incorrect reference labels(17.4%). Notably, 1

