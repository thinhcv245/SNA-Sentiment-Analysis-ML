from lib import load_data
from algorithm.algorithm_training import training_native_bayes, load_model

def training_native_bayes_v(path_output = "sentiment140/processed_training.400000.csv",model_dir = 'model/native_bayes/'):
     # Load data
    data = load_data(path_output)
    '''Tạo model '''
    model, vectorizer = training_native_bayes(data, model_dir)
    
def check_data_native_bayes(new_data):
    model_dir = 'model/native_bayes/'
    dir_model_path = model_dir + 'naive_bayes_model.pkl'
    dir_vectorizer_path = model_dir + 'tfidf_vectorizer.pkl'
    model, vectorizer = load_model(dir_model_path,dir_vectorizer_path)
    new_data_tfidf = vectorizer.transform(new_data)
    predictions = model.predict(new_data_tfidf)
    return predictions 
    # Kết quả sẽ là nhãn 0 hoặc 1

# Main code execution
if __name__ == '__main__':
    new_data = ["I Am Hungry"]
    result = check_data_native_bayes(new_data)

