import pandas as pd

def main():
    nhop_kinship_ds_path = "./data/Kinship/kinship_hinton_qa_nhop.csv" # ALl questions are here
    nhop_kinship_ds_fixed_outpath = "./data/Kinship/kinship_hinton_qna.csv" # ALl questions are here
    nhop_kinship_ds = pd.read_csv(nhop_kinship_ds_path)


    print(f"Columns for dataset are {nhop_kinship_ds.columns}")
    # Take only "Question", "Answer" and "Paths" columnns and drop the rest. 
    nhop_kinship_ds = nhop_kinship_ds[["Question", "Answer", "Paths"]]
    nhop_kinship_ds = nhop_kinship_ds.rename(columns={"Question": "question", "Answer": "answer", "Paths": "triples"})
    nhop_kinship_ds.to_csv(nhop_kinship_ds_fixed_outpath, index=False)
    


if __name__ == "__main__":
    main()
