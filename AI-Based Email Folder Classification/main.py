"""
AI-Based Email Folder Classification System
This application classifies emails into categories: Important, Promotions, Social, Spam, Others
"""

import pandas as pd
import numpy as np
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class EmailClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.categories = ['Important', 'Promotions', 'Social', 'Spam', 'Others']
        
    def preprocess_text(self, text):
        """
        Text preprocessing steps:
        1. Convert to lowercase
        2. Remove punctuation, numbers, special characters
        3. Remove stopwords
        4. Apply tokenization
        5. Apply lemmatization
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation, numbers, and special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        processed_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def load_data(self, file_path):
        """Load email dataset from CSV file"""
        df = pd.read_csv(file_path)
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Preprocess all email texts
        df['Processed_Text'] = df['Email_Text'].apply(self.preprocess_text)
        return df
    
    def train(self, df):
        """Train the classification model"""
        X = df['Processed_Text']
        y = df['Category']
        
        # Split data into training and test sets (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Feature Extraction
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Multinomial Naïve Bayes model
        self.model = MultinomialNB(alpha=1.0)
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, email_text):
        """Predict category for a single email"""
        processed_text = self.preprocess_text(email_text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get probability scores for each category
        probabilities = self.model.predict_proba(text_tfidf)[0]
        prob_dict = {
            category: prob 
            for category, prob in zip(self.model.classes_, probabilities)
        }
        
        return prediction, prob_dict
    
    def predict_batch(self, email_texts):
        """Predict categories for multiple emails"""
        predictions = []
        for text in email_texts:
            pred, prob = self.predict(text)
            predictions.append({'text': text, 'category': pred, 'probabilities': prob})
        return predictions


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="AI Email Folder Classifier",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 AI-Based Email Folder Classification")
    st.markdown("""
    This AI-powered system classifies emails into five categories:
    - 📌 **Important** - Work-related, urgent, personal
    - 🛍️ **Promotions** - Sales, discounts, offers
    - 👥 **Social** - Social media notifications
    - ⚠️ **Spam** - Unwanted, suspicious emails
    - 📄 **Others** - General, miscellaneous emails
    """)
    
    # Initialize classifier
    classifier = EmailClassifier()
    
    # Sidebar for training info
    st.sidebar.header("⚙️ Model Training")
    
    # Load and prepare data
    try:
        df = classifier.load_data('data/emails.csv')
        df = classifier.prepare_data(df)
        
        # Train model
        results = classifier.train(df)
        
        st.sidebar.success(f"✅ Model trained on {len(df)} emails")
        st.sidebar.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        st.sidebar.metric("Precision", f"{results['precision']*100:.2f}%")
        st.sidebar.metric("Recall", f"{results['recall']*100:.2f}%")
        
    except FileNotFoundError:
        st.error("Error: Could not find data/emails.csv. Please ensure the dataset exists.")
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Classify Email", "📊 Model Performance", "📈 Batch Classification"])
    
    with tab1:
        st.header("Classify a Single Email")
        
        # Text input area
        email_text = st.text_area(
            "Paste your email text below:",
            height=150,
            placeholder="Enter email content here..."
        )
        
        # Classify button
        if st.button("Classify Email", type="primary"):
            if email_text.strip():
                prediction, probabilities = classifier.predict(email_text)
                
                st.success(f"### 📁 Predicted Folder: {prediction}")
                
                # Show probability scores
                st.subheader("Confidence Scores:")
                
                # Sort probabilities by value
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                
                # Display as progress bars
                for category, prob in sorted_probs:
                    percentage = prob * 100
                    color = "green" if category == prediction else "blue"
                    st.progress(prob)
                    st.write(f"**{category}**: {percentage:.2f}%")
            else:
                st.warning("Please enter email text to classify.")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision (Weighted)", f"{results['precision']*100:.2f}%")
        with col3:
            st.metric("Recall (Weighted)", f"{results['recall']*100:.2f}%")
        
        # Classification report
        st.subheader("Detailed Classification Report:")
        st.code(results['classification_report'])
        
        # Dataset info
        st.subheader("Dataset Information:")
        st.write(f"Total emails: {len(df)}")
        st.write("\nCategory distribution:")
        category_counts = df['Category'].value_counts()
        st.bar_chart(category_counts)
    
    with tab3:
        st.header("Batch Email Classification")
        st.markdown("Classify multiple emails at once")
        
        # Text area for multiple emails (separated by newlines)
        batch_text = st.text_area(
            "Enter multiple emails (one per line):",
            height=200,
            placeholder="Email 1...\nEmail 2...\nEmail 3..."
        )
        
        if st.button("Classify All", type="primary"):
            if batch_text.strip():
                emails = batch_text.strip().split('\n')
                results_batch = classifier.predict_batch(emails)
                
                st.subheader("Classification Results:")
                
                for i, result in enumerate(results_batch, 1):
                    with st.expander(f"Email {i}: {result['category']}", expanded=True):
                        st.write(f"**Text:** {result['text'][:100]}...")
                        st.write(f"**Category:** {result['category']}")
                        
                        # Show top 3 probabilities
                        sorted_probs = sorted(
                            result['probabilities'].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3]
                        st.write("**Top probabilities:**")
                        for cat, prob in sorted_probs:
                            st.write(f"  - {cat}: {prob*100:.2f}%")
            else:
                st.warning("Please enter emails to classify.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### How it Works:
    1. **Text Preprocessing**: Emails are cleaned (lowercase, remove punctuation/numbers)
    2. **Stopword Removal**: Common words are removed
    3. **Lemmatization**: Words are reduced to their root form
    4. **TF-IDF**: Text converted to numerical features
    5. **Classification**: Multinomial Naïve Bayes classifier predicts the category
    """)


if __name__ == "__main__":
    main()
