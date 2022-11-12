from language_models.preprocessing.preprocessing import ReadSplitDataset, CustomTextDataset
from language_models.preprocessing.vocab_builder import VocabBuilder
from language_models.preprocessing.tokenization import Tokenizer
from language_models.modelling.model import model
from transformers import TrainingArguments, Trainer


def main():
    dataset = ReadSplitDataset("train.csv")
    train_dataset = CustomTextDataset(dataset.x_train, dataset.y_train)
    test_dataset = CustomTextDataset(dataset.x_test, dataset.x_test)
    vocab_builder = VocabBuilder()
    vocabulary = vocab_builder.build_vocab(train_dataset.text)
    tokenizer = Tokenizer()
    train_inputs, train_attention_masks = tokenizer.bert_encode(
        train_dataset.text)

    # Define Trainer
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()
    return train_inputs, train_attention_masks


if __name__ == "__main__":
    train_inputs, train_attention_masks = main()
