import numpy as np
from transformers import BertTokenizer

class Tokenizer:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def bert_encode(self, data):
        input_ids = []
        attention_masks = []

        for index in range(len(data)):
            encoding = self.bert_tokenizer.encode_plus(data[index],
                                                        add_special_tokens=True,
                                                        max_length=512,
                                                        padding='max_length',
                                                        truncation=True,
                                                        return_attention_mask=True)
        
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
            
        return np.array(input_ids), np.array(attention_masks)