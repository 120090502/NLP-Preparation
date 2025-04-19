from tqdm import tqdm
from torch import torch

def translate(self, src_texts, max_length=128, num_beams=5):
    """
    Translate source texts using the model.

    Args:
        src_texts (list of str): Source sentences to translate.
        max_length (int, optional): Maximum length of translations. Defaults to 128.
        num_beams (int, optional): Beam size for beam search. Defaults to 5.

    Returns:
        list of str: Translated sentences.s
    """
    translations = []

    for src_text in tqdm(src_texts[:100]):
        # Tokenize and encode source text
        input_ids = self.tokenizer(
            src_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True
            )

        # Decode translation
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translation)

    return translations