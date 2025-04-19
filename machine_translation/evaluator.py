import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import json
from tqdm import tqdm
import os

class Evaluator:
    def __init__(self, model, tokenizer, device=None):
        """
        Initialize the Evaluator class.

        Args:
            model (torch.nn.Module): The fine-tuned model to evaluate.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for pre/post-processing.
            device (str, optional): The device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def translate(self, src_texts, max_length=128, num_beams=5):
        """
        Translate source texts using the model.

        Args:
            src_texts (list of str): Source sentences to translate.
            max_length (int, optional): Maximum length of translations. Defaults to 128.
            num_beams (int, optional): Beam size for beam search. Defaults to 5.

        Returns:
            list of str: Translated sentences.
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

    def calculate_bleu(self, predictions, references):
        """
        Calculate BLEU score.

        Args:
            predictions (list of str): Model-generated translations.
            references (list of list of str): List of reference translations.

        Returns:
            float: BLEU score.
        """
        bleu = corpus_bleu(predictions, references)
        return bleu.score

    def calculate_rouge(self, predictions, references):
        """
        Calculate ROUGE scores.

        Args:
            predictions (list of str): Model-generated translations.
            references (list of list of str): List of reference translations.

        Returns:
            dict: ROUGE-L, ROUGE-1, and ROUGE-2 scores.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

        for pred, ref in zip(predictions, references):
            scores = scorer.score(pred, ref[0])  # Use the first reference for ROUGE calculation
            rouge_scores["rouge1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge2"] += scores["rouge2"].fmeasure
            rouge_scores["rougeL"] += scores["rougeL"].fmeasure

        # Average over the dataset
        n = len(predictions)
        for key in rouge_scores:
            rouge_scores[key] /= n

        return rouge_scores

    def calculate_bertscore(self, predictions, references, lang="en"):
        """
        Calculate BERTScore.

        Args:
            predictions (list of str): Model-generated translations.
            references (list of list of str): List of reference translations.
            lang (str, optional): Language code (e.g., 'en' for English). Defaults to 'en'.

        Returns:
            dict: Precision, Recall, and F1 scores for BERTScore.
        """
        precision, recall, f1 = bert_score(predictions, [ref[0] for ref in references], lang=lang)
        return {
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item(),
        }

    def evaluate(self, src_texts, references, max_length=128, num_beams=5, case='en-in'):
        """
        Evaluate the model on multiple metrics.

        Args:
            src_texts (list of str): Source sentences to translate.
            references (list of list of str): List of reference translations.
            max_length (int, optional): Maximum length for translations. Defaults to 128.
            num_beams (int, optional): Beam size for beam search. Defaults to 5.
            lang (str, optional): Language code for BERTScore. Defaults to 'en'.

        Returns:
            dict: Dictionary containing BLEU, ROUGE, Perplexity, and BERTScore metrics.
        """
        # Generate translations
        lang = 'id' if 'en' in case else 'jv'
        predictions = self.translate(src_texts, max_length=max_length, num_beams=num_beams)
        print("Finish Predictions")

        # Calculate metrics
        bleu_score = self.calculate_bleu(predictions, references)
        print("Calculated BLUE Score")
        rouge_scores = self.calculate_rouge(predictions, references)
        print("Calculated ROUGE Score")
        bert_scores = self.calculate_bertscore(predictions, references, lang=lang)
        print("Calculated Bert Score")

        results = {
            "BLEU": bleu_score,
            "ROUGE": rouge_scores,
            "BERTScore": bert_scores,
        }

        return predictions, results
    
def process_eval(model_dir, model_name, epoch, test_samples, references, case):

    model_path = os.path.join(model_dir, f'{epoch}.pt')
    if 'T5' in model_path:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    elif 'BART' in model_path:
        tokenizer = MBartTokenizer.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    elif 'Qwen' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Load Model successfully")
    evaluator = Evaluator(model, tokenizer)

    predictions, results = evaluator.evaluate(test_samples, references, max_length=128, num_beams=5, case=case)

    print("Evaluation Results:")
    print(results)

    return predictions, results

def main():
    model_name_list = ['t5-small', 't5-base', 't5-large', 'facebook/mbart-large-50', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B']
    case_list = ['en_in', 'in_ja']
    model_name = "t5-small"
    # base_path = "/root/autodl-tmp/ckpt/T5/small/"
    base_path = "./ckpt/T5/small/"

    for case in case_list:
        path = base_path+case
        # print(os.listdir(base_path))
        test_path = f'./mt/json_data/test_{case}.json'
        with open(test_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)
        test_samples, refer_translation = [item['input'] for item in test_data], [item['output'] for item in test_data]
        
        predictions, evaluate_result = process_eval(model_dir=path, model_name=model_name, epoch=8, test_samples=test_samples, references=refer_translation, case=case)

        eval_file_path = f"./eval/eval_{model_name}_{case}.json"
        perd_file_path = f"./predicitons/eval_{model_name}_{case}.json"

        with open(eval_file_path, 'w') as json_file:
            json.dump(evaluate_result, json_file, indent=4)
        
        with open(perd_file_path, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)       

        print(f"Results saved to {eval_file_path}")
        
if __name__ == "__main__":
    main()