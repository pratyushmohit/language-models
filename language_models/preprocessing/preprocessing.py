import io
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


#custom imports
from .vocab_builder import VocabBuilder
from .tokenization import Tokenizer


class ReadSplitDataset():
    def __init__(self, filepath, test_size=0.2):
        self.filepath = filepath
        self.test_size = test_size
        self.dataset = pd.read_csv(self.filepath, sep="\t", names=['score', 'reviews'])
        self.dataset = self.dataset.head(1000)
        self.x = self.dataset['reviews'].values
        self.y = self.dataset['score'].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, 
                                                                                self.y, 
                                                                                stratify=self.y, 
                                                                                test_size=self.test_size, 
                                                                                shuffle=True)
    
class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.text[index]
        item = {"Text": text, "Class": label}
        return item