import torch


def dataset_router(dataset_parent_path: str) -> Dataset:
    if "freebaseqa" in dataset_parent_path:
        return FreebaseQADataset(dataset_parent_path)
    elif "triviaqa" in dataset_parent_path:
        return TriviaQADataset(dataset_parent_path)
    else:
        raise ValueError("The dataset router could not find a matching dataset for the data path")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, location:str):
        self.location = location

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FreebaseQADataset(Dataset):
    def __init__(self, location:str):
        super().__init__(location)
        # Once such location is loaded, we aim to look fro rth


    def __getitem__(self, idx):
        return self.data[idx]

class TriviaQADataset(Dataset):
    def __init__(self, location):
        super().__init__(location)

    def __getitem__(self, idx):
        return self.data[idx]   

