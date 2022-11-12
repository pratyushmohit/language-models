import json
from transformers import BertConfig, BertForSequenceClassification

class Model:
    def __init__(self, config_file):
    # # Initializing a BERT bert-base-uncased style configuration
        self.config_file = config_file

    def read_config(self):
        self.config = json.load(self.config_file)

    def config(self):
        configuration = BertConfig(vocab_size=30522, 
                            hidden_size=768, 
                            num_hidden_layers=12, 
                            num_attention_heads=12, 
                            intermediate_size=3072, 
                            hidden_act='gelu', 
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12,
                            pad_token_id=0,
                            position_embedding_type='absolute',
                            use_cache=True,
                            classifier_dropout=None)

# # Initializing a model (with random weights) from the bert-base-uncased style configuration
model = BertForSequenceClassification(configuration)