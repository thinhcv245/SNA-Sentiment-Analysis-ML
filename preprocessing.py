import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải stopwords và lemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Hàm tiền xử lý văn bản
def clean_text(text):
    # Chuyển thành chữ thường
    text = text.lower()
    
    # Loại bỏ URL
    text = re.sub(r'http\S+', '', text)
    
    # Loại bỏ các ký tự đặc biệt, số, và dấu câu
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tách từ
    words = text.split()
    
    # Loại bỏ stopwords và lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Gộp lại thành một chuỗi
    return ' '.join(words)

# Áp dụng tiền xử lý cho tất cả văn bản trong dataset
data['cleaned_text'] = data['text'].apply(clean_text)

# Hiển thị vài dòng đầu tiên của dữ liệu đã được xử lý
print(data[['text', 'cleaned_text']].head())
