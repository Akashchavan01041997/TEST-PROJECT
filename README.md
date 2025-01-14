# TEST-PROJECT(project name- prediction of disaster tweet real or fake)
## Disaster Tweet Classification

This project aims to classify tweets related to disasters as real or fake. The notebook `disasterfinal.ipynb` contains the complete workflow from data preprocessing, feature extraction, model training, evaluation, and prediction.

### Installation

To run the notebook and reproduce the results, you need to install the following Python libraries:

```sh
pip install pandas nltk scikit-learn imbalanced-learn gensim textblob
```

### Usage

1. **Data Loading and Preprocessing**: 
   - Load the dataset from a CSV file.
   - Handle missing values using imputation.
   - Preprocess the text data by removing URLs, mentions, non-alphanumeric characters, and stopwords, followed by lemmatization.

2. **Feature Extraction**:
   - Extract features such as text length, number of hashtags, and sentiment polarity using TextBlob.
   - Use pre-trained Word2Vec embedding (`glove-twitter-25`) to convert text into document vectors.

3. **Model Training and Evaluation**:
   - Address class imbalance using SMOTETomek.
   - Split the data into training and testing sets.
   - Train a RandomForestClassifier on the resampled data.
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

4. **Predicting New Tweets**:
   - Preprocess the new tweet.
   - Extract features and convert the text into a document vector.
   - Make a prediction using the trained classifier.

### Example Prediction

```python
new_tweet = "Forest fire near La Ronge Sask. Canada"
processed_tweet = preprocess_text(new_tweet)
tweet_vector = get_document_vector(processed_tweet, model)
text_length = len(new_tweet)
num_hashtags = new_tweet.count('#')
sentiment = TextBlob(new_tweet).sentiment.polarity
new_tweet_combined = np.hstack((tweet_vector, [text_length, num_hashtags, sentiment]))
prediction = classifier.predict([new_tweet_combined])[0]

if prediction == 1:
    print(f"Tweet: '{new_tweet}' is predicted as a real disaster.")
else:
    print(f"Tweet: '{new_tweet}' is predicted as a fake disaster.")
```

