# TEAM

**Team Name:** The Savvy Jokers

**Team Members:**

![team](https://github.com/user-attachments/assets/cf37e54d-9ffd-4edf-9539-02819c688379)

---

# Project: Social Network Analysis (SNA) for Sentiment and Community Detection

## **Description**
The **Social Network Analysis (SNA) for Sentiment and Community Detection** project focuses on analyzing and understanding the structure and dynamics of social networks. This project uses the **Sentiment140** dataset, which contains labeled tweets, to perform sentiment analysis and apply network analysis techniques for detecting communities and understanding emotional trends in social media posts.

We aim to apply various analytical methods, including degree distribution analysis, connected components, clustering coefficients, and centrality measures, to identify key patterns and relationships within the network. The project also integrates sentiment analysis to study emotional responses and interactions in networked environments.

## **Technologies Used**
### Data Analysis & Machine Learning:
- **Python:** ![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
- **Pandas & NumPy:** For data manipulation and analysis.
- **Scikit-learn:** For machine learning and classification tasks.
- **TensorFlow/Keras:** For advanced deep learning models.
- **NLTK & SpaCy:** For Natural Language Processing (NLP) tasks, including sentiment analysis.
- **NetworkX:** For network analysis and visualization.
  
### Libraries & Tools:
- **Matplotlib & Seaborn:** For data visualization.
- **Jupyter Notebooks:** For interactive analysis.
- **BERT & LSTM:** For sentiment analysis and text classification.

---

## **Project Objectives**
The project aims to provide insights into the dynamics of online communities and sentiment within social networks. The key objectives include:

### Functional Requirements:
#### **1. Data Analysis**
- **Network Metrics:** Calculate centrality measures, degree distribution, and clustering coefficients for the social network dataset.
- **Community Detection:** Apply algorithms like Louvain or Girvan-Newman to detect communities within the networks.
- **Path Analysis:** Analyze the shortest paths and network connectivity to understand information flow.

#### **2. Sentiment Analysis**
- **Text Preprocessing:** Tokenization, stopword removal, and vectorization of tweets from the Sentiment140 dataset.
- **Sentiment Classification:** Use machine learning algorithms (Naive Bayes, SVM) and deep learning models (LSTM, BERT) to classify sentiment as positive, negative, or neutral.
- **Sentiment Trends:** Track sentiment shifts over time and correlate with network activity.

#### **3. Visualization & Reporting**
- **Network Visualization:** Visualize the networks and communities using NetworkX and Matplotlib.
- **Sentiment Visualization:** Display sentiment trends using line charts and heatmaps.
- **Dashboard:** Develop a dashboard to present network analysis results and sentiment insights.

### Non-Functional Requirements:
- **Scalability:** Handle large datasets from Sentiment140 efficiently.
- **Reproducibility:** Provide clear documentation and code to ensure that the analysis can be replicated.
- **Interactivity:** Allow users to interact with the visualizations and explore different network metrics.

---

## **Reasons for Choosing the Topic**
- **Understanding Social Media Dynamics:** The need to understand how people interact online and how sentiments evolve in social networks.
- **Community Detection:** Identifying communities can help in analyzing trends, detecting misinformation, or marketing strategies.
- **Sentiment Analysis:** Understanding public sentiment allows businesses and researchers to gain insights into public opinion, customer feedback, and social issues.

---

## **System Architecture**
```plaintext
+-------------------------+      +-----------------------+      +----------------------------+
|       Sentiment140      | ---> |   Data Preprocessing  | ---> |   Sentiment Analysis & ML  |
|      (Tweets Dataset)   |      |   (Cleaning, Token)   |      |  (Naive Bayes, SVM, LSTM)  |
+------------+------------+      +-----------+-----------+      +-------------+--------------+
                                                                              |
                                                                              v
                                                                +----------------------------+
                                                                |      Network Analysis      |
                                                                |   (Community Detection &   |
                                                                |          Metrics)          |
                                                                +-------------+--------------+
                                                                              |
                                                                              v
                                                                      +---------------+
                                                                      | Visualization |
                                                                      |  & Reporting  |
                                                                      +-------+-------+


