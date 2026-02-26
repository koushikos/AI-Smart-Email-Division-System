# AI-Based Email Folder Classification

An AI-powered system that automatically classifies emails into five categories: Important, Promotions, Social, Spam, and Others.

## Features

- 📧 **Email Classification**: Automatically categorizes emails into appropriate folders
- 📊 **Model Performance**: Displays accuracy, precision, and recall metrics
- 🔍 **Confidence Scores**: Shows probability scores for each category
- 📈 **Batch Classification**: Classify multiple emails at once
- 🎨 **User-Friendly Interface**: Built with Streamlit for easy interaction

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
```
bash
pip install -r requirements.txt
```

3. **Run the application:**
```
bash
streamlit run main.py
```

## Project Structure

```
AI-Based Email Folder Classification/
├── data/
│   └── emails.csv          # Sample email dataset
├── main.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── TODO.md               # Implementation plan
```

## How It Works

### 1. Data Collection
- Sample dataset with ~150 emails across 5 categories
- Stored in CSV format with Email_Text and Category columns

### 2. Text Preprocessing
- Convert text to lowercase
- Remove punctuation, numbers, and special characters
- Remove stopwords (common words like "the", "is", etc.)
- Apply tokenization
- Apply lemmatization (reduce words to root form)

### 3. Feature Extraction
- Use TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer
- Converts text into numerical features
- Uses unigrams and bigrams (1-2 word combinations)

### 4. Model Training
- Split data: 80% training, 20% testing
- Uses Multinomial Naïve Bayes classifier
- Evaluates with accuracy, precision, and recall metrics

### 5. Demo Interface
- Streamlit web interface
- Paste email text and click "Classify"
- View predicted folder and confidence scores

## Usage

### Single Email Classification
1. Paste your email text in the text area
2. Click "Classify Email"
3. View the predicted folder and confidence scores

### Batch Classification
1. Go to "Batch Classification" tab
2. Enter multiple emails (one per line)
3. Click "Classify All"
4. View all predictions with confidence scores

## Categories

| Category | Description | Examples |
|----------|-------------|----------|
| Important | Work-related, urgent, personal | Meeting requests, contracts, reminders |
| Promotions | Sales, discounts, offers | Black Friday deals, VIP offers |
| Social | Social media notifications | Friend requests, comments, tags |
| Spam | Unwanted, suspicious emails | Lottery scams, fake prizes |
| Others | General, miscellaneous | Weather updates, cafeteria menus |

## Model Performance

The model typically achieves:
- **Accuracy**: ~90-95% (on sample data)
- **Precision**: High per-class precision
- **Recall**: High per-class recall

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **NLTK**: Natural language processing
- **Streamlit**: Web interface

## Customization

### Adding More Training Data
Edit `data/emails.csv` to add more sample emails:
```
csv
Email_Text,Category
"Your email text here",CategoryName
```

### Changing the Model
Modify the `train()` method in `main.py` to use different classifiers:
```
python
# Logistic Regression
self.model = LogisticRegression(max_iter=1000)

# Random Forest
self.model = RandomForestClassifier(n_estimators=100)
```

## License

This project is for demonstration purposes.

## Author

AI Email Folder Classification Demo
