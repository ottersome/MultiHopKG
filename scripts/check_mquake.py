"""
This file needs to be run at the repos root
"""
import argparse
import json

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_file", "-f", default="./temp/MQuAKE-T.json")

    return ap.parse_args()

def main(args: argparse.Namespace):
    json_file_path: str = args.json_file
    
    # Open and load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    assert isinstance(data, list), "We expect data to be a list"

    # Data is list
    print(f"Data is of type {type(data)}")
    if type(data) is list:
        print(f"Data is of length {len(data)}")
    
    # Show all the keys in the first directory
    print("Keys in the first directory:")
    available_keys = set()
    for elem in data:
        for key in elem.keys():
            available_keys.add(key)
    print("Available keys inside of each element:")
    print(available_keys)

if __name__ == "__main__":
    args = argsies()
    main(args)

