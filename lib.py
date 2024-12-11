import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

def print_data(path):
    data = load_data(path)
    print(data.head())  # Display the first few rows of the dataset
