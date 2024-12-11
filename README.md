# Sentiment140 - Phân tích cảm xúc

Dự án này sử dụng bộ dữ liệu Sentiment140 để thực hiện phân tích cảm xúc. Nó sử dụng môi trường ảo Python để quản lý các thư viện và phụ thuộc.

## Hướng dẫn cài đặt môi trường ảo
````
python -m venv data_preprocessing
````
``` python
data_preprocessing\Scripts\activate
```
e   # Windows
# source data_preprocessing/bin/activate  # macOS/Linux

# Cài đặt các thư viện yêu cầu:
```
pip install -r requirements.txt
```
# Chạy theo thứ tự 
Số 1: Tải dữ liệu
```
python dowloat_datasets.py
```
Số 2: Tách dữ liệu
```
python get400kdata.py

```
# Hủy môi trường
```
deactivate
```

#  Tạo requirements.txt
```
pip freeze > requirements.txt
```


# Thực thi lệnh nltk.download
chạy môi trường ảo
```
data_preprocessing\Scripts\activate
```
chạy tiếp
```
python
```
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

Thoát
```
exit()
```
