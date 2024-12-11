import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Download necessary NLTK resources

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """Load data from CSV file, with an optional header"""
    try:
        data = pd.read_csv(file_path, encoding='latin-1')
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {e}")
        return None
def save_data(data, file_path):
    """Save processed data to a CSV file"""
    try:
        data.to_csv(file_path, index=False, encoding='latin-1')
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Preprocessing Functions
def remove_URL(text):
    """Remove URLs from the text"""
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    """Remove emojis from the text"""
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    """Remove HTML tags from the text"""
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    """Remove punctuation from the text"""
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_non_alpha(text):
    """Remove non-alphabetic characters"""
    return re.sub(r'[^a-zA-Z\s]', '', text)

def clean_text(text):
    """Apply all preprocessing functions to the text"""
    text = text.lower()  # Convert text to lowercase
    text = remove_URL(text)  # Remove URLs
    text = remove_emoji(text)  # Remove emojis
    text = remove_html(text)  # Remove HTML tags
    text = remove_punct(text)  # Remove punctuation
    text = remove_non_alpha(text)  # Remove non-alphabetic characters
    words = text.split()  # Split the text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(words)  # Join words back into a string