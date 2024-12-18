import pandas as pd
import joblib
from lib import load_data
from algorithm.algorithm_training import training_native_bayes, load_model,training_SVM,training_SVM_V2

def training_native_bayes_v(path_output = "sentiment140/processed_training.1600000.csv",model_dir = 'model/native_bayes/'):
     # Load data
    data = load_data(path_output)
    '''Tạo model '''
    model, vectorizer = training_native_bayes(data, model_dir)
    return model
    
def training_SVM_v(path_output = "sentiment140/processed_training.1600000.csv",model_dir = 'model/SVM/'):
     # Load data
    data = load_data(path_output)
    '''Tạo model '''
    model = training_SVM(data, model_dir)
    return model
def training_SVM_v2_update(path_output = "sentiment140/processed_training.1600000.csv",model_dir = 'model/SVM/'):
     # Load data
    print("Loading training..")
    data = load_data(path_output)
    print("Loading training done..")

    '''Tạo model '''
    model = training_SVM_V2(data, model_dir)
    return model
def check_data_SVM(new_data):
    model_dir = 'model/SVM/'
    dir_model_path = model_dir + 'svm_model.pkl'
    dir_vectorizer_path = model_dir + 'tfidf_vectorizer.pkl'
    model, vectorizer = load_model(dir_model_path, dir_vectorizer_path)
    pca = joblib.load(model_dir + 'pca.pkl' ) 

    if isinstance(new_data, list):
        new_data = pd.Series(new_data)
    new_data = new_data.fillna('').astype(str)
    new_data_tfidf = vectorizer.transform(new_data)
    new_data_tfidf_pca = pca.transform(new_data_tfidf.toarray())  # Giảm số chiều cho dữ liệu mới

    predictions = model.predict(new_data_tfidf_pca)
    return predictions

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
    
    
    model = training_native_bayes_v()
    # new_data = ["I love this product!", "This is the worst experience I’ve had."]
    # result = check_data_native_bayes(new_data)
    # print(result)
    #training_SVM_v()
    #training_logistic_regression_v()
    #training_native_bayes_v(path_output = "sentiment140/processed_training.1600000_2.csv",model_dir = 'model/native_bayes/')

    training_SVM_v2_update(path_output = "sentiment140/processed_training.1600000_2.csv",model_dir = 'model/SVM/')
    # result = check_data_SVM(new_data)
    # print(result)