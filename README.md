# SPRING2024_PROJECT_MODELCARD_ARTIFICIALINTELLIGENCE_FOR_DIGITAL_TRANSFORMATION

# Business case
The objective is to develop an ML model to predict the outcome of the 2024 U.S. presidential election based on sentiment analysis of publicly available text data that in form of Tweets. 
The model would help political analysts, campaign managers, news agencies, and bettors understand public sentiment and potential voting patterns.
Insights derived from the model could inform campaign strategies, media coverage, political advertizers and financial markets.

# Business value of using ML (impact)

The goal is to leverage ML for predicting election outcomes, enabling businesses to make informed investment decisions in advertising and campaign strategies.
Provides political strategists and campaigners with insights into public opinion trends, enabling data-driven decision-making.
Forecasts election outcomes, aiding in investment decisions related to campaign financing and media planning.
Maximizing ROI: By improving the accuracy of sentiment analysis, the model aids in minimizing losses and risks associated with political investments, optimizing tax strategies, and maximizing ROI.

# ML framing

**Project Archetype : Sentiment Analysis and Stance Detection in Political Context**

•	AI-2024-US-Election-Sentiment-Analyzer Goal: Adapt and enhance the existing 2020 election sentiment analysis model for the 2024 US Elections.
•	Model Description Model Name: AI-2024-US-Election-Sentiment-Analyzer 
•	Model Type: Sentiment Analysis using pre-trained BERT. 
•	Primary Use: Analysis of public sentiment in tweets related to US Elections 2024
•	Language: English
•	Region of Application: United States (Not applicable for Morocco)
•	Architecture: Based on BERT (Bidirectional Encoder Representations from Transformers)
•	Training Dataset: Over 10 million English tweets regarding US political discussions, specifically filtered for US Elections 2024 topics.
•	Target Audience: Political analysts, campaign strategists, digital marketers, and researchers
•	Use Cases: Predicting election outcomes, analyzing public opinion trends, enhancing targeted political campaigning.

**Data Requirements**

A large and diverse dataset of text data, including social media posts, news articles, and transcripts of political speeches and debates.
The data should cover a wide range of political topics, demographics, and geographical locations to ensure representativeness.
Data should be annotated with reliable sentiment labels, either manually or through a semi-automated process with human oversight.
Access to historical election data, polling results, and demographic information for model training and contextual understanding.



**Feasibility including specification of a baseline**
The baselines are NLP models on sentiment analysis tasks, such as SVM or Naive Bayes classifiers, before the introduction of BERT. 
For details about the baseline and the datasets please consult:

https://huggingface.co/kornosk
https://github.com/GU-DataLab/stance-detection-KE-MLM
https://aclanthology.org/2021.naacl-main.376/

**Providing arguments for my choice (e.g., published state of the art results, give credit to the authors)**

The following paper made me think to improve the used method and then made such a choice. This paper "Knowledge Enhanced Masked Language Model for Stance Detection" by Kornraphop Kawintiranon and Lisa Singh presented a novel approach to improve stance detection in tweets related to political entities, specifically in the context of the 2020 US Presidential election. 
https://aclanthology.org/2021.naacl-main.376/
Both the binary of the trained model and the notebook to retrain are available.

Methdology:

Problem Definition: They define stance detection as classifying text to determine if the position is in support, opposition, or neutral towards a target, such as a political candidate.
Challenges of Twitter Data: The authors acknowledge the unique challenges posed by Twitter's data, including the brevity of tweets, the rapid evolution of language and terminology, and the deviation from standard prose.
Fine-Tuning Language Models: They discuss how fine-tuning pre-trained models with large-scale in-domain data has become the state-of-the-art approach for many NLP tasks, including stance detection.
Weighted Log-Odds-Ratio for Word Identification: Unlike random token masking in traditional BERT pre-training, the authors propose using weighted log-odds-ratio to identify and focus on words with high stance distinguishability. This helps in better understanding the stance each word represents in the context of political discourse.
Attention Mechanism Focusing on Key Words: By modeling an attention mechanism that concentrates on stance-significant words, the language model becomes more attuned to stance-related nuances in the text.
Performance Superiority: They show that their proposed approach outperforms the state-of-the-art in stance detection for Twitter data about the 2020 US Presidential election.
Methodology:
  They collect and preprocess a dataset of tweets related to the election.
  Significant stance words are identified using knowledge mining techniques.
  An enhanced Masked Language Model (MLM) is created by masking these significant tokens.
  The model is fine-tuned on the stance detection task using a labeled dataset.
  Their novel KE-MLM approach integrates these significant tokens during the fine-tuning process for BERT.
Empirical Evaluation: They evaluate their model using a dataset containing labeled tweets about the election, showing improvements in detection accuracy.
Contributions and Resources: The authors release their labeled stance dataset to aid further research and highlight their contributions, including the log-odds-ratio technique for identifying stance words and the novel fine-tuning method.
Conclusions and Future Work: They conclude that their approach can serve as a new standard for stance detection. They also suggest future research directions, such as adjusting the significance level of tokens during fine-tuning and exploring the application of their method to other NLP tasks.

Sentiment Analysis:
The goal of sentiment analysis is to determine the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions, and emotions expressed within an online mention.
It is primarily concerned with identifying the polarity of text content - whether the expressed opinion in the text is positive, negative, or neutral.
Sentiment analysis is often applied to social media posts, reviews, and any other places where people express their opinions to understand consumer sentiment, public opinion, etc.
Techniques include machine learning models such as Naive Bayes, Logistic Regression, SVM, and deep learning approaches like LSTM, as well as lexicon-based methods that assign sentiment scores to words.
Stance Detection:
Stance detection is about determining the writer's position (favor, against, neutral) on a specific topic or claim, regardless of the emotional content.
Unlike sentiment analysis, which focuses on the affective aspect, stance detection identifies the alignment of the author's point of view with a particular standpoint.
It is a crucial task in understanding public opinion on controversial issues, gauging support for political candidates or policies, and analyzing debates.
Stance detection often requires understanding the context beyond the text itself, as a positive sentiment does not necessarily mean support for a certain stance, and vice versa.
The challenge in stance detection lies in the subtleties of language, as the same expression can be used sarcastically or earnestly, and the stance might not be explicitly stated but implied.

Both tasks are important for different reasons and use cases:
Sentiment analysis is widely used in brand monitoring, market research, and customer service, as it provides insights into how people feel about products, services, or topics.
Stance detection is particularly useful in political science, public policy, and argument mining, where understanding the position someone takes is more critical than the emotional tone of their language.
Advanced NLP systems may perform both tasks together to gain a comprehensive understanding of text data, using sentiment analysis to capture the emotional tone and stance detection to understand the position towards the subject matter.


**Model card for the baseline, after getting inspired from Huggingface model cards**

Pre-trained BERT on Twitter US Election 2020 for Stance Detection towards Joe Biden (KE-MLM)
Pre-trained weights for KE-MLM model in Knowledge Enhance Masked Language Model for Stance Detection, NAACL 2021.

Training Data
This model is pre-trained on over 5 million English tweets about the 2020 US Presidential Election. Then fine-tuned using our stance-labeled data for stance detection towards Joe Biden.

Training Objective
This model is initialized with BERT-base and trained with normal MLM objective with classification layer fine-tuned for stance detection towards Joe Biden.

Usage
This pre-trained language model is fine-tuned to the stance detection task specifically for Joe Biden.

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



**Datasets are uploaded in the same branch of this GitHub Link, other Datasets will be WebScrapped using a specific tool : TwExtract**

**Metrics for business goal evaluation**
Accuracy of the model in predicting election outcomes compared to actual results.
Accuracy of Stance Detection, Organization Satisfaction
Precision and recall in classifying sentiments towards specific candidates or issues.
Timeliness of insights, as political campaigns require rapid response to changing public opinions.
Robustness of the model against shifts in political discourse and the introduction of new topics or candidates.
Improvement Over Baselines: The degree to which the new model outperforms existing models and baselines could be a business metric, indicating the value added by the new approach.

**Other References**
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9530657
https://www.researchgate.net/publication/354427394_Biden_vs_Trump_Modelling_US_general_elections_using_BERT_language_model
https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden-KE-MLM
https://paperswithcode.com/paper/knowledge-enhanced-masked-language-model-for

