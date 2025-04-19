import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from util import Util
from trainer import Trainer

if __name__ == "__main__":
    # Load model and tokenizer
    # version_list = ['small', 'base', 'large']
    version = 'small'
    model_name = f"t5-{version}"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # We perfom these two translation tasks Seprately
    case_list = ['en_in', 'in_ja']

    for case in case_list:
        if case=='en-in':
            src_lang, tgt_lang = "english", "indonesian"
        else: 
            src_lang, tgt_lang = "indonesian", "javanese"
        
        # Prepare data
        util = Util(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer=tokenizer,
            add_prompt=f"Please translate the following sentence from {src_lang.capitalize()} to {tgt_lang.capitalize()}: ",
        )
        train_dataloader, val_dataloader, test_dataloader = util._dataloader(batchsize=16)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            save_dir=f"./ckpt/T5/{version}",
            device="cuda",
            learning_rate=5e-5,
            scheduler_type="cosine",
            num_epochs=20,
        )

        # Train the model
        trainer.train()