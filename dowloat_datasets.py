#Tải datasets về
import kagglehub
import shutil

def download_datasets():
    # Download latest version
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print("Path to dataset files:", path)
    return path

def move_datasets(path, target_path ):
    shutil.move(path, target_path)
    print(f"Dataset moved to: {target_path}")

def AoutoDowloadDatasets():
    path = download_datasets()
    #di chuyển dataset đến thư mục 
    target_path = "./sentiment140"
    move_datasets(path, target_path)
    print("Download and move datasets completed successfully.")
#main script
if __name__ == "__main__":
    #tải dataset
    path = download_datasets()
    #di chuyển dataset đến thư mục 
    target_path = "./sentiment140"
    move_datasets(path, target_path)
    print("Download and move datasets completed successfully.")

