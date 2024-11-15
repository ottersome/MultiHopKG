import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import os

    
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def split_data(data, train_size, val_size, test_size, random_seed):
    # Ensure the sizes sum to 1
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1."
    # Split data into train and temp (val + test)
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_seed)
    
    # Calculate the proportion of validation data in the temp data
    val_proportion = val_size / (val_size + test_size)
    
    # Split temp data into validation and test
    val_data, test_data = train_test_split(temp_data, train_size=val_proportion, random_state=random_seed)
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Train-Validation-Test Split for a .txt file.')
    parser.add_argument('--file_path', type=str, default="./data/itl/triplets_fb_wiki_v2.txttriplets_fj_wiki.txt", help='Path to the input .txt file.')
    parser.add_argument('--train_size', type=float, default=0.98, help='Proportion of data to use for training.')
    parser.add_argument('--val_size', type=float, default=0.01, help='Proportion of data to use for validation.')
    parser.add_argument('--test_size', type=float, default=0.01, help='Proportion of data to use for testing.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    dirname = os.path.dirname(args.file_path)
    print(f"Working directory here is {dirname}")
    train_path = os.path.join(dirname, "train.triples")
    dev_path = os.path.join(dirname, "dev.triples")
    test_path = os.path.join(dirname, "test.triples")

    # Load data from the file
    data = load_data(args.file_path)
    # Perform the split
    train_data, val_data, test_data = split_data(data, args.train_size, args.val_size, args.test_size, args.random_seed)
    # Output the results
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")
    # Optionally, save the splits to separate files
    with open(train_path, 'w') as f:
        print(f"Saving {len(train_data)} samples of train data in : {train_path}")
        f.writelines(train_data)
    with open(dev_path, 'w') as f:
        print(f"Saving {len(val_data)} samples of val data in : {dev_path}")
        f.writelines(val_data)
    with open(test_path, 'w') as f:
        print(f"Saving {len(test_data)} samples of test data in : {test_path}")
        f.writelines(test_data)

    # For good measure copy same data to raw.csv l
    raw_kb_dir = os.path.join(dirname, "raw.kb")
    with open(raw_kb_dir, 'w') as f: 
        print(f"For good measure I'll also be saving the training data in this directory {raw_kb_dir}")
        f.writelines(train_data)

if __name__ == '__main__':
    main()
