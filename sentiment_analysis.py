import pandas as pd
import spacy


# Spacy model
nlp = spacy.load('en_core_web_md')

# Function to preprocess the text data
def preprocess_text(dataframe):
    clean_data = dataframe.dropna(subset=['reviews.text'])
    return clean_data

# Function for sentiment analysis
def analyse_sentiment(review):
    doc = nlp(review)
    sentiment_score = doc.sentiment
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to compare the similarity of two product reviews
def compare_similarity(review1, review2):
    doc1 = nlp(review1)
    doc2 = nlp(review2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

# Function to remove stop words
def remove_stop_words(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(tokens)


dataset_path = r'C:\Users\tgarn\Downloads\Python\1\Task 21\amazon_product_reviews.csv'

data = pd.read_csv(dataset_path)

# Preprocess the text data
clean_data = preprocess_text(data)

# Iterate through each review and print its sentiment
for review in clean_data['reviews.text'].iloc[0:2]:
    sentiment = analyse_sentiment(review)
    print("Review:", review)
    print("Sentiment:", sentiment)
    print()

# Test of compare_similarity function 
review1 = clean_data['reviews.text'].iloc[0]
review2 = clean_data['reviews.text'].iloc[1]
similarity_score = compare_similarity(review1, review2)
print("Similarity score between review 1 and review 2:", similarity_score)
print ()

# Remove stop words from review1 and review2
clean_review1 = remove_stop_words(review1)
clean_review2 = remove_stop_words(review2)

# Print cleaned reviews
print("Review 1 with stop words removed is:", clean_review1)
print("Review 2 with stop words removed is:", clean_review2)

