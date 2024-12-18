from services import *

#region library
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Core packages for text processing.
import string
import re
# Libraries for text preprocessing.
import nltk
# nltk.download('punkt_tab', download_dir='E:\\SNA-Sentiment-Analysis-ML\\venv\\nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Loading some sklearn packaces for modelling.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import f1_score, accuracy_score

# Some packages for word clouds and NER.
from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image
# import spacy
# import en_core_web_sm

# Core packages for general use throughout the notebook.

import random
import warnings
import time
import datetime

# For customizing our plots.
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Setting some options for general use.
stop = set(stopwords.words('english'))
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')

#endregion

# === 1. Đọc và xử lý dữ liệu ===
# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, "data/raw/sentiment140.csv"), encoding='latin-1', header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

df['text_clean'] = df['text'].apply(lambda x: remove_URL(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_emoji(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_html(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_punct(x))

df['tokenized'] = df['text_clean'].apply(word_tokenize)
df.to_csv(os.path.join(base_dir, 'data', 'raw', 'sentiment140_processed.csv'))