import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from util import Util
from trainer import Trainer

if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "facebook/mbart-large-50"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Prepare data
    util = Util(
        src_lang="english",
        tgt_lang="indonesian",
        tokenizer=tokenizer,
        add_prompt="Please translate the following sentence from English to Indonesian: ",
    )

    train_dataloader, val_dataloader, test_dataloader = util._dataloader(batchsize=16)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        save_dir="./ckpt/mBART/",
        device="cuda",
        learning_rate=5e-5,
        scheduler_type="cosine",
        num_epochs=10,
    )

    # Train the model
    trainer.train()