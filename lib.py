import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Download necessary NLTK resources

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """Load data from CSV file, with an optional header"""
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {e}")
        return None
def save_data(data, file_path):
    """Save processed data to a CSV file"""
    data.to_csv(file_path, index=False)
    print(f"File đã được tạo và lưu tại {file_path}")


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

def expand_contractions(text):
    contractions_dict = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'t": " not", "'ve": " have", "'m": " am"
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def replace_slang(text):
    slang_dict = {
        "u": "you", "r": "are", "luv": "love", "idk": "i don't know", 
        "btw": "by the way", "omg": "oh my god", "ttyl": "talk to you later", 
        "bff": "best friend forever", "brb": "be right back", "gr8": "great", 
        "thx": "thanks", "ty": "thank you", "np": "no problem", "imo": "in my opinion", 
        "imho": "in my humble opinion", "lmao": "laughing my ass off", 
        "rofl": "rolling on the floor laughing", "smh": "shaking my head", 
        "afaik": "as far as I know", "fyi": "for your information", "asap": "as soon as possible", 
        "jk": "just kidding", "tbh": "to be honest", "idc": "i don't care", 
        "ikr": "i know right", "ily": "i love you", "dw": "don't worry", 
        "nvm": "never mind", "gtg": "got to go", "wtf": "what the f***", 
        "lol": "laugh out loud", "smh": "shaking my head", "afk": "away from keyboard", 
        "lmk": "let me know", "wyd": "what you doing", "fml": "f*** my life", 
        "yo": "hello", "slay": "do well", "fam": "family", "bffl": "best friends for life", 
        "bae": "before anyone else", "bruh": "bro", "lit": "awesome", "srsly": "seriously", 
        "wtb": "want to buy", "lmao": "laugh my ass off", "roflmao": "rolling on the floor laughing my ass off", 
        "ppl": "people", "stfu": "shut the f*** up", "fomo": "fear of missing out", 
        "yolo": "you only live once", "bored af": "bored as f***", "np": "no problem", 
        "omw": "on my way", "tmi": "too much information", "ikr": "I know right", 
        "ily": "i love you", "gr8": "great", "ttfn": "ta-ta for now", "lil": "little", 
        "nope": "no", "yup": "yes", "smexy": "sexy", "fyi": "for your information", 
        "fwb": "friends with benefits", "slay": "do something really well", 
        "bffl": "best friends for life", "holla": "hello", "snatched": "looks good", 
        "ratchet": "someone or something that is unattractive or unrefined", "turnt": "excited, hyped, or energetic", 
        "on fleek": "perfectly done", "tbh": "to be honest", "savage": "tough, ruthless, or impressive", 
        "lit": "amazing, exciting", "lowkey": "subtle, not obvious", "highkey": "obvious, in your face", 
        "vibe": "mood or atmosphere", "cray": "crazy", "woke": "socially aware or enlightened",
        "sksksk": "expression of excitement or laughter", "and I oop": "expression of surprise or shock"
    }
    words = text.split()
    return ' '.join([slang_dict[word] if word in slang_dict else word for word in words])

def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_text = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_text)


def remove_mentions(text):
    """Remove mentions starting with '@' from the text"""
    return re.sub(r'@\w+', '', text)

def remove_repeated_chars(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


def clean_text(text):
    if not text:  # Kiểm tra nếu text là None hoặc rỗng
        return ""
    text = text.lower()  # Chuyển thành chữ thường
    text = remove_URL(text)  # Loại bỏ URL
    text = expand_contractions(text)  # Mở rộng từ viết tắt
    text = remove_mentions(text)  # Loại bỏ đề cập
    text = remove_emoji(text)  # Loại bỏ emoji
    text = remove_html(text)  # Loại bỏ HTML
    text = remove_punct(text)  # Loại bỏ dấu câu
    text = remove_non_alpha(text)  # Loại bỏ ký tự không phải chữ
    text = replace_slang(text)  # Thay thế từ lóng
    text = remove_repeated_chars(text)  # Loại bỏ ký tự lặp lại
    text = remove_extra_whitespace(text)  # Loại bỏ khoảng trắng thừa

    # Xử lý lemmatization và stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)

    # Kiểm tra lỗi chính tả
    # text = correct_spelling(text)

    return text
