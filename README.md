
# PROJECT NAME : US ELECTIONS 2024 RESULTS PREDICTION

### SPRING2024_PROJECT_MODELCARD_ARTIFICIALINTELLIGENCE_FOR_DIGITAL_TRANSFORMATION
### By : Adnane El Amrani

## Table of Contents
1. [Business Case](#business-case)
2. [Business Value of Using ML (Impact)](#business-value-of-using-ml-impact)
3. [ML Framing](#ml-framing)
4. [Data Requirements](#data-requirements)
5. [Feasibility and Baseline](#feasibility-including-specification-of-a-baseline)
6. [Choice Justification](#providing-arguments-for-my-choice)
7. [Paper Methodology](#paper-methodology)
8. [Sentiment Analysis](#sentiment-analysis)
9. [Stance Detection](#stance-detection)
10. [Importance of Both Tasks](#importance-of-both-tasks)
11. [Model Card for Baseline](#model-card-for-the-baseline-after-getting-inspired-from-huggingface-model-cards)
12. [Sample Code](#sample-code)
13. [Datasets and Tools](#datasets-are-uploaded-in-the-same-branch-of-this-github-link-other-datasets-will-be-webscrapped-using-a-specific-tool-twextract)
14. [Metrics for Business Goal Evaluation](#metrics-for-business-goal-evaluation)
15. [Other References](#other-references)

## Business Case

The objective is to develop an ML model to predict the outcome of the 2024 U.S. presidential election based on sentiment analysis of publicly available text data that in form of Tweets. 
The model would help political analysts, campaign managers, news agencies, and bettors understand public sentiment and potential voting patterns.
Insights derived from the model could inform campaign strategies, media coverage, political advertizers and financial markets.

## Business Value of Using ML (Impact)

The goal is to leverage ML for predicting election outcomes, enabling businesses to make informed investment decisions in advertising and campaign strategies.
Provides political strategists and campaigners with insights into public opinion trends, enabling data-driven decision-making.
Forecasts election outcomes, aiding in investment decisions related to campaign financing and media planning.
Maximizing ROI: By improving the accuracy of sentiment analysis, the model aids in minimizing losses and risks associated with political investments, optimizing tax strategies, and maximizing ROI.

## ML Framing

***Project Archetype: Sentiment Analysis and Stance Detection in Political Context***
- **Model Name:** AI-2024-US-Election-Sentiment-Analyzer
- **Goal:** Adapt and enhance the existing 2020 election sentiment analysis model for the 2024 US Elections.
- **Model Type:** Sentiment Analysis using pre-trained BERT.
- **Primary Use:** Analysis of public sentiment in tweets related to US Elections 2024.
- **Language:** English.
- **Region of Application:** United States (Not applicable for Morocco).
- **Architecture:** Based on BERT (Bidirectional Encoder Representations from Transformers).
- **Training Dataset:** Over 10 million English tweets regarding US political discussions, specifically filtered for US Elections 2024 topics.
- **Target Audience:** Political analysts, campaign strategists, digital marketers, and researchers.
- **Use Cases:** Predicting election outcomes, analyzing public opinion trends, enhancing targeted political campaigning.


## Data Requirements

- **Source Diversity:** A large and diverse dataset of text data from Twitter, encompassing social media posts, news articles, and transcripts of political speeches and debates.
- **Broad Coverage:** The data should cover a wide range of political topics, demographics, and geographical locations to ensure representativeness.
- **Sentiment Labeling:** Data should be annotated with reliable sentiment labels, either manually or through a semi-automated process with human oversight.
- **Historical and Contextual Data:** Access to historical election data, polling results, and demographic information for model training and contextual understanding.


## Feasibility including specification of a baseline

![image](https://github.com/elamraniadnane1/SP24/assets/46249118/fdbd04f5-2243-4fec-8e35-4c7235da4a1b)

## Sentiment Analysis Overview

### Deep Learning Methods
- **BERT:** Falls under the "Deep Learning" category within "Sentiment Analysis", considered alongside neural network-based approaches such as:
  - CNN (Convolutional Neural Network)
  - RNN (Recurrent Neural Network)
  - LSTM (Long Short-Term Memory)
- **Applications:** BERT is suitable for sentiment analysis and can potentially be used for social network analysis to understand communication contexts within a network.

### Volumetric Analysis
- **Description:** Predicts election outcomes based on the volume of mentions or activity related to a candidate or party, assuming that more mentions correlate with higher popularity or potential electoral success.

### Sentiment Analysis Approaches
- **Process:** Computational determination of whether text is positive, negative, or neutral, used to gauge public opinion from text data on various platforms.

### Lexicon-Based Methods
- **Dictionary:** Simple form where words are tagged as positive or negative.
- **Statistical:** Uses statistical techniques to assign sentiment scores based on word co-occurrence frequencies.
- **Sentiword:** Likely refers to SentiWordNet, assigning sentiment scores of positivity, negativity, and objectivity to each WordNet synset.

### Machine Learning Methods
- **Supervised Learning:** Predicts sentiment of new texts using labeled datasets.
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Decision Tree
- **Unsupervised Learning:** Finds patterns without pre-labeled responses.
  - K-Means
  - KNN (k-Nearest Neighbors)

### Deep Learning Methods
- **CNN (Convolutional Neural Network):** Used for image recognition and sentence classification in NLP.
- **RNN (Recurrent Neural Network):** Suitable for sequence prediction due to its internal state (memory).
- **LSTM (Long Short-Term Memory):** Capable of learning long-term dependencies, effective for sequences in tasks like time series analysis or NLP.

### Social Network Analysis
- **Definition:** Uses network and graph theories to understand social structures, such as how information spreads through networks and the influence of various actors within a social network for election prediction.

- **Baseline Models:** The baselines are NLP models on sentiment analysis tasks, such as SVM or Naive Bayes classifiers, before the introduction of BERT.
- **Additional Resources and Details:**
  - For more details about the baseline and its datasets, please consult the following links:
    - [Hugging Face - kornosk](https://huggingface.co/kornosk)
    - [GitHub - GU-DataLab/stance-detection-KE-MLM](https://github.com/GU-DataLab/stance-detection-KE-MLM)
    - [ACL Anthology - Knowledge Enhanced Masked Language Model for Stance Detection](https://aclanthology.org/2021.naacl-main.376/)


## Choice Justification

The following paper inspired me to improve the used method and consequently make such a choice. The paper, ["Knowledge Enhanced Masked Language Model for Stance Detection"](https://aclanthology.org/2021.naacl-main.376/) by Kornraphop Kawintiranon and Lisa Singh, presented a novel approach to improve stance detection in tweets related to political entities, specifically in the context of the 2020 US Presidential election. Both the binary of the trained model and the notebook to retrain are available.

## Paper Methodology:

- **Problem Definition:** They define stance detection as classifying text to determine if the position is in support, opposition, or neutral towards a target, such as a political candidate.
- **Challenges of Twitter Data:** The authors acknowledge the unique challenges posed by Twitter's data, including the brevity of tweets, the rapid evolution of language and terminology, and the deviation from standard prose.
- **Fine-Tuning Language Models:** They discuss how fine-tuning pre-trained models with large-scale in-domain data has become the state-of-the-art approach for many NLP tasks, including stance detection.
- **Weighted Log-Odds-Ratio for Word Identification:** Unlike random token masking in traditional BERT pre-training, the authors propose using weighted log-odds-ratio to identify and focus on words with high stance distinguishability. This helps in better understanding the stance each word represents in the context of political discourse.
- **Attention Mechanism Focusing on Key Words:** By modeling an attention mechanism that concentrates on stance-significant words, the language model becomes more attuned to stance-related nuances in the text.
- **Performance Superiority:** They show that their proposed approach outperforms the state-of-the-art in stance detection for Twitter data about the 2020 US Presidential election.

#### Steps in their Methodology:
1. Collect and preprocess a dataset of tweets related to the election.
2. Identify significant stance words using knowledge mining techniques.
3. Create an enhanced Masked Language Model (MLM) by masking these significant tokens.
4. Fine-tune the model on the stance detection task using a labeled dataset.
5. Integrate these significant tokens during the fine-tuning process for BERT in their novel KE-MLM approach.

- **Empirical Evaluation:** They evaluate their model using a dataset containing labeled tweets about the election, showing improvements in detection accuracy.
- **Contributions and Resources:** The authors release their labeled stance dataset to aid further research and highlight their contributions, including the log-odds-ratio technique for identifying stance words and the novel fine-tuning method.
- **Conclusions and Future Work:** They conclude that their approach can serve as a new standard for stance detection. They also suggest future research directions, such as adjusting the significance level of tokens during fine-tuning and exploring the application of their method to other NLP tasks.

## Sentiment Analysis:

- **Goal:** The goal of sentiment analysis is to determine the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions, and emotions expressed within an online mention.
- **Concern:** It is primarily concerned with identifying the polarity of text content - whether the expressed opinion in the text is positive, negative, or neutral.
- **Applications:** Sentiment analysis is often applied to social media posts, reviews, and any other places where people express their opinions to understand consumer sentiment, public opinion, etc.
- **Techniques:** Techniques include machine learning models such as Naive Bayes, Logistic Regression, SVM, and deep learning approaches like LSTM, as well as lexicon-based methods that assign sentiment scores to words.

## Stance Detection:

- **Objective:** Stance detection is about determining the writer's position (favor, against, neutral) on a specific topic or claim, regardless of the emotional content.
- **Focus:** Unlike sentiment analysis, which focuses on the affective aspect, stance detection identifies the alignment of the author's point of view with a particular standpoint.
- **Importance:** It is a crucial task in understanding public opinion on controversial issues, gauging support for political candidates or policies, and analyzing debates.
- **Challenges:** The challenge in stance detection lies in the subtleties of language, as the same expression can be used sarcastically or earnestly, and the stance might not be explicitly stated but implied.

## Importance of Both Tasks:

- **Sentiment Analysis Uses:** Sentiment analysis is widely used in brand monitoring, market research, and customer service, as it provides insights into how people feel about products, services, or topics.
- **Stance Detection Uses:** Stance detection is particularly useful in political science, public policy, and argument mining, where understanding the position someone takes is more critical than the emotional tone of their language.
- **Combined Application:** Advanced NLP systems may perform both tasks together to gain a comprehensive understanding of text data, using sentiment analysis to capture the emotional tone and stance detection to understand the position towards the subject matter.


## Model Card for the Baseline

Inspired by Huggingface model cards, this section provides details about the pre-trained BERT model used as the baseline for stance detection in the context of the 2020 US Presidential Election.

### Model Overview
- **Model Name:** KE-MLM (Knowledge Enhanced Masked Language Model)
- **Source Paper:** "Knowledge Enhance Masked Language Model for Stance Detection, NAACL 2021"

### Training Data
- **Dataset Size:** The model is pre-trained on over 5 million English tweets.
- **Context:** The tweets pertain to the 2020 US Presidential Election.
- **Fine-Tuning:** Further refined with stance-labeled data specifically for detecting stance towards Joe Biden.

### Training Objective
- **Initialization:** BERT-base is utilized as the starting point.
- **MLM Objective:** Trained with the normal MLM (Masked Language Model) objective.
- **Fine-Tuning Detail:** The classification layer is fine-tuned for stance detection towards Joe Biden.

### Usage
- **Application:** This language model is specifically fine-tuned for the stance detection task concerning Joe Biden.

### Accessing the Model
- **Model Files:** To access the model files, visit the Hugging Face repository at [kornosk/bert-election2020-twitter-stance-biden](https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden/tree/main).


## Sample Code 
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select mode path here
pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)

id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}

##### Prediction Neutral #####
sentence = "Hello World."
inputs = tokenizer(sentence.lower(), return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

##### Prediction Favor #####
sentence = "Go Go Biden!!!"
inputs = tokenizer(sentence.lower(), return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

##### Prediction Against #####
sentence = "Biden is the worst."
inputs = tokenizer(sentence.lower(), return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Against:", predicted_probability[0])
print("Favor:", predicted_probability[1])
print("Neutral:", predicted_probability[2])

```

## Datasets are uploaded in the same branch of this GitHub Link, other Datasets will be WebScrapped using a specific tool : TwExtract**

## Metrics for Business Goal Evaluation

- **Accuracy of Election Outcome Prediction:** The model's accuracy in predicting election outcomes compared to actual results.
- **Stance Detection Accuracy and Organization Satisfaction:** How accurately the model detects stances, and the satisfaction level of organizations using it.
- **Precision and Recall in Sentiment Classification:** The model's precision and recall in classifying sentiments towards specific candidates or issues.
- **Timeliness of Insights:** The ability of the model to provide rapid insights, crucial for political campaigns responding to changing public opinions.
- **Robustness Against Political Discourse Shifts:** The model's robustness against shifts in political discourse and the introduction of new topics or candidates.
- **Improvement Over Baselines:** The degree to which the new model outperforms existing models and baselines, indicating the added value of the new approach.


## Other References

- [IEEE Xplore - [Paper Title or Description]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9530657)
- [ResearchGate - Biden vs. Trump: Modelling US General Elections using BERT Language Model](https://www.researchgate.net/publication/354427394_Biden_vs_Trump_Modelling_US_general_elections_using_BERT_language_model)
- [Hugging Face Model Repository - kornosk/bert-election2020-twitter-stance-biden-KE-MLM](https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden-KE-MLM)
- [Papers With Code - Knowledge-Enhanced Masked Language Model for Stance Detection](https://paperswithcode.com/paper/knowledge-enhanced-masked-language-model-for)
- [Retraining a BERT Model for Transfer Learning in Requirements Engineering: A Preliminary Study](https://www.researchgate.net/publication/365112611_Retraining_a_BERT_Model_for_Transfer_Learning_in_Requirements_Engineering_A_Preliminary_Study)

