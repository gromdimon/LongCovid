# Analysis of Long COVID Symptoms using BERT and Twitter Data
![header](images/header.jpg)


## Table of Contents
1. Introduction
2. Getting Started
   - Prerequisites
   - Installation
3. Data Collection
4. Preprocessing
5. BERT Model
   - Model Architecture
   - Training
   - Evaluation
6. Results and Analysis
7. Conclusion
8. Contributors
9. References


## 1. Introduction
Long COVID refers to a range of symptoms that persist for weeks or months after the acute phase of a COVID-19 infection. Analyzing and understanding these symptoms is crucial for effective management and treatment. In this project, we leverage the power of BERT (Bidirectional Encoder Representations from Transformers) and Twitter data to perform an analysis of Long COVID symptoms. By utilizing BERT's contextual word embeddings and Twitter's rich dataset, we aim to identify and gain insights into the symptoms experienced by individuals with Long COVID.


## 2. Getting Started
### Prerequisites
To run the notebooks from that project, you need the following prerequisites:
- Python 3.8 or higher
- PyTorch (deep learning library)
- Transformers (Hugging Face library for BERT models)
- Pandas (data manipulation library)
- Numpy (numerical computing library)
- Matplotlib (data visualization library)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/long-covid-analysis.git
   ```

2. Install the required libraries using pip:
   ```bash
   pip install tensorflow tweepy transformers pandas numpy matplotlib
   ```

### Alternative Installation
If you are using Google Colab, you can run the following notebooks directly from the repository:
- [Data Collection]()
- [Preprocessing]()
- [BERT Model]()
- [Results and Analysis]()


## 3. Data Collection
In this step, we collected data related to Long COVID symptoms from multiple sources, including Twitter and various articles from newspapers. The Twitter data was collected accessing the Twitter API, while the articles were provided to us. 

### Twitter Data
To collect Twitter data, we used appropriate keywords and hashtags related to Long COVID symptoms to search for relevant tweets. The data collected from Twitter includes metadata such as the timestamp of the tweet, user information, hashtags used, number of retweets, and other engagement metrics. 

However, it is important to note that Twitter data comes with certain challenges and considerations:

#### Retweets and Duplicates
Retweets are a common occurrence on Twitter, where users repost or share someone else's tweet. This can lead to duplicate data, as the same content may appear multiple times. To address this, we implemented a mechanism to remove retweets and identify unique tweets for analysis. 

#### Noisy and Irrelevant Data
Twitter data can contain noise, such as spam, advertisements, or unrelated content. During the data collection process, we applied filters to remove irrelevant tweets and focus only on those specifically related to Long COVID symptoms. This helped ensure the quality and relevance of the collected data.

### Article Data
Alongside Twitter data, we also obtained basic data from various articles provided to us. These articles were sourced from newspapers and other reliable publications. The article data includes metadata such as publication date, author information, article title, and other relevant information. This additional data complements the Twitter data and provides a broader perspective on Long COVID symptoms.


## 4. Preprocessing
Before performing analysis, we conducted preprocessing steps to clean and prepare the collected data. This included both the Twitter data and the newspaper articles, which were analyzed together.

### Data Cleaning
For the combined dataset, we applied data cleaning techniques to remove noise, handle special characters, and standardize the text format. This involved removing irrelevant content, such as spam, advertisements, and unrelated information, to ensure the data was focused specifically on Long COVID symptoms.

### Duplicate Removal
To eliminate duplicate entries and ensure each piece of content was unique, we implemented a duplicate removal mechanism. This process helped maintain data integrity and avoid biased analysis due to duplicated information.

### Stopword Removal
In order to identify specific and relevant words in the dataset, we employed a stopwords dataset. Stopwords are commonly used words that do not carry significant meaning in the context of analysis. By removing stopwords, we aimed to focus on the most relevant terms related to Long COVID symptoms in both the Twitter data and the newspaper articles.

### Data Labeling
To facilitate supervised learning and enable training of our BERT model, we performed data labeling. The combined dataset was labeled using the [LabelStudio](https://labelstud.io/) tool, which provides an interface for annotating and labeling text data. This labeling process involved assigning relevant symptom labels to the data, ensuring the accuracy and consistency of the labeled dataset.

By conducting comprehensive preprocessing on the combined Twitter data and newspaper articles, we prepared the dataset for further analysis. These steps ensured the data was cleaned, duplicates were removed, irrelevant words were filtered out, and the data was labeled for training our BERT model to identify Long COVID symptoms effectively.


## 5. BERT Model
### Model Architecture
The BERT model is a deep transformer-based model that can learn powerful word representations from large amounts of unlabeled data. It consists of an encoder stack with multiple layers, attention mechanisms, and self-attention mechanisms. The BERT model's pre-trained weights are used as a starting point, and fine-tuning is performed on the Long COVID dataset.

### Training
The preprocessed Long COVID dataset is split into training, validation, and testing sets. The BERT model is trained using the training set with appropriate hyperparameters, such as learning rate, batch size, and number of epochs. During training, the model's performance is monitored on the validation set to avoid overfitting.

### Evaluation
After training, the performance of the BERT model is evaluated on the testing set. Evaluation metrics such as accuracy, precision, recall, and F1 score are calculated to assess the model's effectiveness in identifying Long COVID symptoms from Twitter data. Additionally, qualitative analysis can be performed to gain insights into the most frequent symptoms and their distribution.


## 6. Results and Analysis
The results obtained from the BERT model are presented in this section. It includes performance metrics, such as accuracy, precision, recall, and F1 score, along with visualizations to illustrate the distribution of Long COVID symptoms identified from the Twitter data. Furthermore, analysis and interpretation of the results are provided to understand the prevalence and patterns of Long COVID symptoms.

## 7. Conclusion
In

 this project, we utilized BERT and Twitter data to perform an analysis of Long COVID symptoms. By training a BERT model on a labeled Long COVID dataset, we successfully identified and analyzed symptoms from Twitter data. The results obtained provide valuable insights into the prevalence and patterns of Long COVID symptoms. This analysis can contribute to the understanding and management of Long COVID, assisting healthcare professionals in providing appropriate care to affected individuals.


## 8. Contributors
This project was conducted as part of the "Deep Learning, practical approach" course at Freie Universität Berlin ([FU-Berlin](https://www.fu-berlin.de/)). The project team consisted of the following contributors:

- Dzmitry Hramyka
- Kristin Öhlek
- Nirmal Sasthankuttypillai

### Supervisors
The project was supervised by:

- Vitaly Belik
- Andrzej Jarynowski
- Alexander Semenov

Each team member made significant contributions to various aspects of the project, including data collection, preprocessing, model training, analysis, and result interpretation. The collaborative effort and individual contributions from all team members were essential for the successful completion of this project.

We would like to express our sincere appreciation to the instructors and teaching staff of the "Deep Learning, practical approach" course at FU-Berlin for their guidance, support, and valuable feedback throughout the project. Their expertise and knowledge greatly enriched the project and contributed to its success.


## 9. References
Include a list of references or resources that were used during the project, such as academic papers, official documentation, or relevant websites.

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- Hugging Face Transformers Library - [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- Pandas Documentation - [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- Numpy Documentation - [https://numpy.org/doc/](https://numpy.org/doc/)
- Matplotlib Documentation - [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- Seaborn Documentation - [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html)
- LabelStudio Documentation - [https://labelstud.io/](https://labelstud.io/)
- Kaggle - [https://www.kaggle.com/](https://www.kaggle.com/)
- [BERT for "Everyone"](https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation/notebook) Kaggle notebook by [Harsh Jaoin](https://www.kaggle.com/harshjain123
- Header Image - [CureVac](https://www.curevac.com/en/covid-19/)