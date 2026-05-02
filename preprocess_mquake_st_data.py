from typing import List, Tuple
import pandas as pd
import argparse
import os

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mquake_data_dir', default="./data/mquake/")
    ap.add_argument('--mquake_output_dir', default="./data/mquake_salesforce_compatible/")

    return ap.parse_args()

def main():
    args = get_args()

    # Make sure our target directory exists:
    os.makedirs(args.mquake_output_dir, exist_ok=True)

    # Start with a clean target output dir
    os.system(f"rm -rf {args.mquake_output_dir}/*")
    print(f"Cleaned {args.mquake_output_dir}")

    triples_formation(args.mquake_data_dir, args.mquake_output_dir)
    create_raw_kb(args.mquake_output_dir)


def triples_formation(input_dir:str, output_dir: str):
    # Will form .* triples files out of old files. 
    # Read each file line by line and reorder columns from 1-2-3 to 1-3-2

    target_split = ['train', 'dev', 'test']
    for i,in_split in enumerate(['train', 'valid', 'test']):
        input_path = os.path.join(input_dir, f"{in_split}.txt")
        output_path = os.path.join(output_dir, f"{target_split[i]}.triples")
        
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if line:
                    columns = line.split()
                    if len(columns) == 3:
                        # Reorder from 1-2-3 to 1-3-2
                        reordered_line = f"{columns[0]}\t{columns[2]}\t{columns[1]}\n"
                        outfile.write(reordered_line)
        
        print(f"Processed {input_path} to {output_path} with reordered columns")


def create_raw_kb(output_dir: str):
    # Create raw.kb file containing union of all triples in 1-3-2 format
    all_triples = set()
    
    # Read all triples from train, dev, and test files
    # for split in ['train', 'dev', 'test']:
    # Apparently its just the train directory.
    for split in ['train']:
        triples_path = os.path.join(output_dir, f"{split}.triples")
        if not os.path.exists(triples_path):
            print("You need to run triplets_formation, before this function")
            exit(-1)
        with open(triples_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_triples.add(line)
    
    print(f"About to write {len(all_triples)} triples")
    # Write union of all triples to raw.kb
    raw_kb_path = os.path.join(output_dir, 'raw.kb')
    with open(raw_kb_path, 'w') as f:
        for triple in sorted(all_triples):
            f.write(f"{triple}\n")
    
    print(f"Created raw.kb with {len(all_triples)} unique triples")

                            
if __name__ == "__main__":

    main()
