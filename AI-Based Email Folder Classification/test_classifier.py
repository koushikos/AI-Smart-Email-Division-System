"""Test script for the Email Classifier"""
import pandas as pd
from main import EmailClassifier

# Initialize classifier
classifier = EmailClassifier()

# Load data
df = classifier.load_data('data/emails.csv')
print(f'Loaded {len(df)} emails')

# Prepare data
df = classifier.prepare_data(df)
print('Data preprocessing completed')

# Train model
results = classifier.train(df)
print(f'Accuracy: {results["accuracy"]*100:.2f}%')
print(f'Precision: {results["precision"]*100:.2f}%')
print(f'Recall: {results["recall"]*100:.2f}%')

# Test predictions
test_emails = [
    'Urgent meeting scheduled for tomorrow at 2 PM',
    'Limited time offer - 50% off all items!',
    'You have been tagged in a photo by John',
    'Congratulations you have won a free iPhone!',
    'The weather forecast for tomorrow is sunny'
]

print('\n=== Test Predictions ===')
for email in test_emails:
    pred, probs = classifier.predict(email)
    print(f'Email: {email}')
    print(f'Predicted Category: {pred}')
    # Show top 2 probabilities
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
    for cat, prob in sorted_probs:
        print(f'  {cat}: {prob*100:.2f}%')
    print()
