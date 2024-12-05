from torch.utils.data import Dataset
import pandas as pd

class CNN_dataset(Dataset):
    def __init__(self, dataframe, config, tokenizer, model_name="gpt2"):
        self.dataframe = dataframe.dropna()
        self.config = config
        self.model_name = model_name
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int):
        article = self.tokenizer(self.dataframe.iloc[idx]["article"], padding="max_length", return_tensors="pt", truncation=True)
        label = self.tokenizer(self.dataframe.iloc[idx]["highlights"], padding="max_length", return_tensors="pt", truncation=True)
        label_text = self.dataframe.iloc[idx]["highlights"]
        return article, label, label_text
    
    def __len__(self):
        return len(self.dataframe)
    

