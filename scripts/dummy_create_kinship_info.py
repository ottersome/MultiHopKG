import pandas as pd

entities_dict_path = "./data/Kinship/entities.dict"
relations_dict_path = "./data/Kinship/relations.dict"
entitiesinfo_dict_outpath = "./data/Kinship/entities_info.csv"
relationsinfo_dict_outpath = "./data/Kinship/relations_info.csv"

entities_csv = pd.read_csv(entities_dict_path, header=None, sep="\s")
relations_csv = pd.read_csv(relations_dict_path, header=None, sep="\s")

entities_info = pd.DataFrame()
entities_info["QID"] = entities_csv.iloc[:,1]
entities_info["Title"] = entities_csv.iloc[:,1]
entities_info.to_csv(entitiesinfo_dict_outpath, index=False)


# Likewise
relations_info = pd.DataFrame()
relations_info["QID"] = relations_csv.iloc[:,1]
relations_info["Title"] = relations_csv.iloc[:, 1]
relations_info.to_csv(relationsinfo_dict_outpath, index=False)

