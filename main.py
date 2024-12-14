from lib import load_data, clean_text, save_data
from get400kdata import getdata
from dowloat_datasets import AoutoDowloadDatasets, download_datasets, download_datasets_elon_musk_tweets
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


    #path = download_datasets_elon_musk_tweets()
    #AoutoDowloadDatasets(path)
    path_data = "sentiment140/336/elon_musk_tweets.csv"

    path_output = "sentiment140/elon_musk_tweets_dataset.csv"
    '''Chạy để lấy data và triển khai clean data'''

    data = load_data(path_data)
    clear_data_save_data(data, path_output)
    print_data(path_output)
    # data2 = getdata(path_output)
    # print(data2.head())
    # data = load_data(path_output)
    #print_data(path_data)