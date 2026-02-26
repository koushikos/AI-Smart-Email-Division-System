"""Standalone test for Email Classifier without Streamlit imports"""
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(processed_tokens)

# Load data
print("Loading data...")
df = pd.read_csv('data/emails.csv')
print(f"Loaded {len(df)} emails")

# Prepare data
print("Preprocessing text...")
df['Processed_Text'] = df['Email_Text'].apply(preprocess_text)

# Train model
print("Training model...")
X = df['Processed_Text']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"\n=== Model Performance ===")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")

# Test predictions
print("\n=== Test Predictions ===")
test_emails = [
    'Urgent meeting scheduled for tomorrow at 2 PM',
    'Limited time offer - 50% off all items!',
    'You have been tagged in a photo by John',
    'Congratulations you have won a free iPhone!',
    'The weather forecast for tomorrow is sunny'
]

for email in test_emails:
    processed = preprocess_text(email)
    text_tfidf = vectorizer.transform([processed])
    pred = model.predict(text_tfidf)[0]
    probs = model.predict_proba(text_tfidf)[0]
    prob_dict = {cat: prob for cat, prob in zip(model.classes_, probs)}
    
    print(f"\nEmail: {email}")
    print(f"Predicted: {pred}")
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:2]
    for cat, prob in sorted_probs:
        print(f"  {cat}: {prob*100:.2f}%")

print("\n=== TEST COMPLETED SUCCESSFULLY ===")
