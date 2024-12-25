# config.py
DATA_PATH = "./data/sentiment140.csv"
PROCESSED_PATH = "./data/processed.csv"

# Cấu hình mô hình DepIC
ALPHA = 0.2        # Cường độ kích hoạt
BETA_POS = 0.3     # Cường độ cảm xúc tích cực
BETA_NEG = 0.2     # Cường độ cảm xúc tiêu cực
TIME_DECAY = 0.003 # Tham số suy giảm thời gian
MAX_STEPS = 5     # Số bước lan truyền tối đa
