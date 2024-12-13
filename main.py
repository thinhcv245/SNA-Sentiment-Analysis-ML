from lib import load_data, clean_text, save_data
from get400kdata import getdata
from dowloat_datasets import AoutoDowloadDatasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def print_data(path):
    data = load_data(path)
    print(data.head())  # Display the first few rows of the dataset
def clear_data_save_data(data, path_output):
    if data is not None:
        # Preprocess text
        data['clean_text'] = data['text'].apply(clean_text)
        # Save the processed data
        save_data(data, path_output)
    else:
        print("Data loading failed.")
        
# Main code execution
if __name__ == '__main__':
    # Load data


    AoutoDowloadDatasets()
    '''Chạy để lấy data và triển khai clean data'''
    path_data = "sentiment140/training.1600000.processed.noemoticon.csv"
    path_output = "sentiment140/processed_training.1600000.csv"
    data = getdata(path_data)
    clear_data_save_data(data, path_output)

    # data2 = getdata(path_output)
    # print(data2.head())
    # data = load_data(path_output)
    #print_data(path_data)