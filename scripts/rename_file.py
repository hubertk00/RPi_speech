import os
import argparse

def rename_files(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1] 
        new_name = f"{i}{ext}"  
        os.rename(os.path.join(path, filename), os.path.join(path, new_name)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change name of files in a folder")
    parser.add_argument('--path', type=str, required=True, 
                        help="Path to folder to rename")
    
    args = parser.parse_args()

    rename_files(args.path)