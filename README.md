# bulgarian_reviews-sentiment-analysis
Sentiment analysis on movies' reviews from Bulgaria

This project presents a binary sentiment classifier built on movie reviews and compares the traditional baseline models to a fine-tuned multilingual transformers.

# Overview

The baseline is implemented as a standard ML pipeline combining TF-IDF vectorization and a linear classifier(Logistic Regression/ SVM), while the choice for fine-tuned multilingual transformer is mDeBERTa-v3. 

The primary objective is to classify user-generated reviews as positive or negative, addressing inherent challenges such as ambiguous language, class imbalance and noisy annotation, and systematically compare the performance between interpretable classical ML models and black-box deep learning models.

# Dataset
The experiments use publicly available dataset, containing bulgarian-language user-generated reviews. The dataset is hosted on Hugging Face(https://huggingface.co/datasets/DGurgurov/bulgarian_sa). 
The dataset comes into three splits:

|Split          |        Samples|
|-------------- |-------------- |
| Training      |          5,412|
| Validation    |            838|
| Test          |          1,673|

An initial examination of the training subset revealed a substantial class imbalance in favor of the positive reviews as follows:

|Review           |        Samples|
|---------------- |-------------- |
| Positive(1)     |          4,587|
| Negative(0)     |            825|

This imbalace motivated the use of class-weighted training and macro averaged F1-score as trustworthy metrics for the purpose of this project. 

# Methodology

## Baseline models
Two traditional ML classifiers were evaluated: Logistic Regression(LR) and Linear Support Vector Machine (SVM). 

Both models were implemented using scikit-learn pipelines combining TF-IDF vectorization of character n-grams with a linear classifier. Text preprocessing consisted of lowercasting, trimming leading and trailing whitespace, and normalizing consecutive whitespace characters. 

Hyperparameter selection was performed using grid search with stratified 5-fold cross-validation, optimizing macro-averaged F1-score. The search included TF-IDF n-gram configurations, regularization strenght, and class weights. Following hyperparameter selection, out-of-fold predictions were generated from the training set to optimize the classification decision threshold with macro-F1 as the objective. 

The selected configurations were: 

|Hyperparameters                           |LR               |SVM             |
|------------------------------------------|---------------- |--------------- |
|**ngram_range**                           |(2,7)            |(1,7)           |
|**C**                                     |3.0              |0.3             | 
|**class weight{0, 1}**                    |{0: 3.0, 1: 1.0} |{0: 3.0, 1: 1.0}|
|**predicted probability/decision score**  |0.54             | 0.01           |

## Transformer Fine-Tuning
The second approach differs as it involvs a pretrained multilingual transofrmer is performed. 
Unlike TF-IDF representation, transformer model generate contextual contextual representations in which token representations deend on the surrounding text. 

Fine-tuning was performed using Pytorch. The configuration included tuning the **learning rate**, **number of training epochs**, **weight_decay** again with the objective to maximize average f1_macro. The selected hyperparameters were:

|Hyperparameters          |value          |
|------------------------ |-------------- |
| **learning rate**       |2e-05          |
| **num_train_epochs**    |3.0            |
| **weight decay**        |0.01           |

# Experimental Results(from the testing fold)

## Baseline 

|Model        | Accuracy | Macro-F1 | Macro Precision | Macro Recall|
|-----------  |--------- |----------| ----------------| ------------|
| LR          |0.9331    |0.8966    | 0.8958          | 0.8975      |
| Linear SVM  |0.9366    |0.9002    | 0.9077          | 0.8932      |
| mDeBERTa-v3 |0.9588    |0.9342    | 0.9487          | 0.9213      |


Class specific results: 

|model                    |LR     |       |SVM    |       | mDeBERTa-v3|         |
--------------------------|-------|-------|-------|-------|------------|---------|        
| Class                   |0      |1      |0      |1      |0           |1        |
| Precision               |0.8328 |0.9573 |0.8607 |0.9548 |0.9327      |0.9647   |
| Recall                  |0.8378 |0.9573 |0.8201 |0.9663 |0.8584      |0.9843   |
| F1                      |0.8353 |0.9580 |0.8399 |0.9605 |0.8940      |0.9744   |

Confusion matrices 

1. LR
<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/98003a50-d851-45fe-86f7-6cf9981b400e" />

2. SVM
<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/a3254a6f-e7ec-489c-bcdc-4c9dd7f23d17" />

3. mDeBERTa - v3
<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/cdee090d-a1ed-4353-90bd-78d691b29903" />

The overall accuracy for both baseline models exceeded 93% but the breakdown by class revealed the stark performance disparity between the two classes. This underlines the importance of class-specific metrics and macro-f1 when handling class imbalance data. Of course, there are specific techniques that could tackle such obstacles but given the colorful language used by users in movie reviews, a cautious and well-thought through approach is advised when applying methods to address class imbalance problems. :) 

# Error Analysis

The manual inspection of false-positives and false-negatives from the baseline models shed light to the limitations of such feature extraction method - the context blindness. This is a known limitation of the TF-IDF features. 

Since the errors from the transformer moder were only 69, this allowed manual review for each error. The nature of the error was subjectively categorized. According to the results the predominant categories were mixed sentiment and insufficient context accounting for 53.6% of the errors, followed by general model limitation(23.2%) and potentially incorrect reference labels(17.4%). Notably, 15 of the 16 cases classified as general model limitation were false positives. 

# Future Improvement
1. Ensemble models
The LR and Linear SVM classifiers exhibited different error distributions. While SVM generated more false positives, LR produced more false negatives. Although both models used TF-IDF character n-gram representations and identicall class weights, they employed different selected feature configuration, resulting in different learned decision boundaries. Hence a potential direction for further investigation on the baseline is ensemble learning. An initial experiment could measure the overlap between the misclassified examples to determine whether the models make siffuciently complementary errors. Then probability calibration and soft voting could be explored in order to improve the classification performance. 

2. Emoji-Aware Feature Engineering 
EDA suggesed, that the emoji usage is more frequently in a combination with negative negation, so maybe Emoji-Aware TF-IDF could be examined as possible extension.


# Licence 
This project is licensed under the MIT License.

