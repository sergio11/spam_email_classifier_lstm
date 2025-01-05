# Spam Email Classification using LSTM 📧🤖

This project explores building a classification model to differentiate between **Spam** and **Legitimate (Ham)** emails using **Long Short-Term Memory (LSTM)** networks. The notebook details the end-to-end process of preparing the data, training the model, and evaluating its performance.

The focus is on using **Natural Language Processing (NLP)** techniques for text preprocessing and **Deep Learning** to classify emails based on their content. By the end of the project, we aim to have a trained model that can effectively predict whether an email is spam or legitimate.

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

We would like to express our gratitude to **purusinghvi** for creating and sharing the **Spam Email Classification Dataset - Combined Spam Email CSV of 2007 TREC Public Spam Corpus and Enron-Spam Dataset** on Kaggle. This dataset, which contains detailed information about spam and legitimate emails, has been invaluable in building and training the machine learning model for spam detection.

🌟 The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset). Your contribution is greatly appreciated! 🙌

## 🌟 Explore My Other Deep Learning Projects! 🌟  

If you found this project intriguing, I invite you to check out my other cutting-edge deep learning initiatives:  

### [🌍 Advanced Classification of Disaster-Related Tweets Using Deep Learning 🚨](https://github.com/sergio11/disasters_prediction)  
How does social media respond to crises in real time? This project focuses on **classifying tweets** to determine if they’re related to disasters or not. Using **deep learning** and enriched datasets, I uncover insights into how people discuss disasters on platforms like Twitter.  

### [📰 Fighting Misinformation: Source-Based Fake News Classification 🕵️‍♂️](https://github.com/sergio11/fake_news_classifier)  
In a world plagued by fake news, this project tackles the challenge of **authenticity in journalism**. Using **source-based analysis** and **machine learning**, I aim to classify news articles as real or fake, promoting a more informed and responsible media ecosystem.

### [💳 Fraud Detection Model with Deep Neural Networks (DNN)](https://github.com/sergio11/online_payment_fraud)  
Fraudulent transactions can cause immense financial losses. This project leverages **Deep Neural Networks** to detect fraud in financial data, even in highly imbalanced datasets. Learn about my process, from **Exploratory Data Analysis (EDA)** to building a **scalable and efficient solution** for fraud detection.

### [🛡️ IoT Network Malware Classifier with Deep Learning Neural Network Architecture 🚀](https://github.com/sergio11/iot_network_malware_classifier)  
The Internet of Things (IoT) is transforming the world, but it also introduces new security challenges. This project leverages **Deep Learning Neural Networks** to classify and detect malware in IoT network traffic. By analyzing patterns with AI, it provides **proactive cybersecurity solutions** to safeguard interconnected devices. Explore the intricate process of model design and training with the **Keras framework**, detailed in the accompanying Jupyter Notebook.

Take a dive into these projects to see how **deep learning** is solving real-world problems and shaping the future of AI applications. Let's innovate together! 🚀

## Key Steps in the Process 🛠️

### 1. **Data Collection & Preprocessing 📊**
- **Loading the Data**: The dataset consists of emails labeled as **Spam (1)** or **Legitimate (0)**.
- **Text Normalization**: We start by converting text to lowercase and removing unnecessary characters, such as numbers, punctuation, and special symbols.
- **Stopword Removal**: Common words that do not contribute to meaningful classification (like "the", "and", etc.) are removed.
- **Hyperlink Removal**: URLs and hyperlinks in the text are deleted as they do not provide useful information for classification.
- **Tokenization**: We split the email text into individual words (tokens) for easier processing.

### 2. **Exploratory Data Analysis (EDA) 🔍**
- **Visualizing the Data**: The notebook includes visualizations such as word clouds and n-gram analysis, which help in understanding the most common terms used in spam and legitimate emails.
- **Class Distribution**: The dataset is explored to understand the distribution of spam vs. legitimate emails, which helps in deciding model evaluation strategies.

### 3. **Feature Engineering ⚙️**
- **Text Tokenization**: The email text is tokenized into sequences, and the vocabulary is built.
- **Padding**: As the text data varies in length, padding is applied to ensure that all input sequences have the same size, making them suitable for model input.
- **Label Encoding**: The target labels (spam or legitimate) are encoded into numeric values (0 or 1) using LabelEncoder.

### 4. **Model Construction 🏗️**
- **Bi-directional LSTM**: We use a **Bi-directional LSTM** model to process the sequence of words in both forward and backward directions. This helps capture contextual information from both past and future words.
- **Dense Layer**: A fully connected layer with **ReLU** activation is added to capture non-linear relationships between features.
- **Dropout**: A dropout layer is included to prevent overfitting and help the model generalize better.

### 5. **Model Training 🚀**
- The model is trained on the preprocessed training data using **binary cross-entropy loss** and the **Adam optimizer**.
- **Early Stopping** is implemented to monitor the validation loss and stop training once the model starts overfitting.
- **Evaluation**: The model is evaluated on a separate test set to determine its accuracy and ability to generalize to unseen data.

### 6. **Model Evaluation and Results 📊**
- **Training Metrics**: The model's performance during training is tracked by monitoring the loss and accuracy.
- **Validation Metrics**: The validation loss and accuracy provide insight into how well the model generalizes.
- **Overfitting**: If the validation accuracy starts to drop while training accuracy continues to rise, it indicates overfitting. This is addressed by using techniques like **early stopping**.

## Goals of the Project 🎯
- To classify emails as **Spam** or **Legitimate** using deep learning.
- To explore NLP techniques for text preprocessing and sequence classification.
- To evaluate the model's performance on both training and validation sets, and improve it through strategies like **early stopping** and **dropout**.

## Results 📈
- The model’s performance on the training data is typically high, with **99%** accuracy.
- On the validation data, accuracy usually reaches around **97%**, though slight fluctuations are observed due to overfitting.

## Conclusion 🎓
By the end of this project, you will have a functional **Bi-LSTM model** for spam email classification that can be further fine-tuned, deployed, or integrated into a larger system for filtering unwanted emails. Techniques like **early stopping** are crucial to prevent overfitting and ensure the model’s generalizability.

## **📚 References**
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [NLP with Disaster Tweets Challenge](https://www.kaggle.com/c/nlp-getting-started)
- [https://www.kaggle.com/code/hrhuynguyen/detecting-spam-in-emails-with-lstms-99-accuracy](https://www.kaggle.com/code/hrhuynguyen/detecting-spam-in-emails-with-lstms-99-accuracy)
- [https://www.kaggle.com/code/rem4000/spam-email-xgboost-99](https://www.kaggle.com/code/rem4000/spam-email-xgboost-99)

## **🙏 Acknowledgments**

A huge **thank you** to **purusinghvi** for providing the dataset that made this project possible! 🌟 The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset). Your contribution is greatly appreciated! 🙌

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

## Visitors Count

<img width="auto" src="https://profile-counter.glitch.me/spam_email_classifier_lstm/count.svg" />

## Please Share & Star the repository to keep me motivated.
<a href = "https://github.com/sergio11/spam_email_classifier_lstm/stargazers">
   <img src = "https://img.shields.io/github/stars/sergio11/spam_email_classifier_lstm" />
</a>

## License ⚖️

This project is licensed under the MIT License, an open-source software license that allows developers to freely use, copy, modify, and distribute the software. 🛠️ This includes use in both personal and commercial projects, with the only requirement being that the original copyright notice is retained. 📄

Please note the following limitations:

- The software is provided "as is", without any warranties, express or implied. 🚫🛡️
- If you distribute the software, whether in original or modified form, you must include the original copyright notice and license. 📑
- The license allows for commercial use, but you cannot claim ownership over the software itself. 🏷️

The goal of this license is to maximize freedom for developers while maintaining recognition for the original creators.

```
MIT License

Copyright (c) 2024 Dream software - Sergio Sánchez 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

