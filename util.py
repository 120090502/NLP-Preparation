import json
import pandas as pd
from torch.utils.data import DataLoader
from openai import OpenAI
import torch
import os

class TranslationDataset():

    def __init__(self, src_texts, tgt_texts, tokenizer, add_prompt, max_length=128):
        """_summary_

        Args:
            src_texts (List[str]): Source Language dataset to be translated
            tgt_texts (List[str]): Target Language dataset (true label for translation)
            tokenizer (class): Tokenizer Class
            add_prompt (str): Added instruction prompt, default to be ''
            max_length (int, optional): Max Seq Length for the model training. Defaults to 128.
        """

        self.src_texts = [add_prompt + text for text in src_texts]
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_encoding = self.tokenizer(src_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        tgt_encoding = self.tokenizer(tgt_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        return {
            'input_ids': src_encoding['input_ids'].squeeze(0),
            'attention_mask': src_encoding['attention_mask'].squeeze(0),
            'labels': tgt_encoding['input_ids'].squeeze(0)
        }


class Util():
    def __init__(self, src_lang, tgt_lang, tokenizer, add_prompt='', use_both=False):
        """_summary_

        Args:
            src_texts (List[str]): Source Language dataset to be translated
            tgt_texts (List[str]): Target Language dataset (true label for translation)
            tokenizer (class): Tokenizer
            add_prompt (str, optional): Added prompt added in the beginning of the beginning of every training corpus. Defaults to ''.
            use_both (bool, optional): Use both languaged Dataset if set to be True. Defaults to False.
        """
        self.src = src_lang
        self.tgt = tgt_lang
        self.base_path = "./mt/"
        self.tokenizer = tokenizer
        self.add_prompt = add_prompt
        self.use_both = use_both
        print(self.src)

    def _load_json_file(self, file_path):
        """
        Load a JSON file from the given path.
        Returns the parsed JSON data as a Python dictionary or list.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

    def _convert2json(self, save_dir='./mt/'):
        scenario = ['train', 'test', 'valid']
        for sce in scenario:
            cur_data = pd.read_csv(self.base_path + sce + ".csv")
            src_en, tgt_in = cur_data['english'], cur_data['indonesian']
            src_in, tgt_ja = cur_data['indonesian'], cur_data['javanese']
            
            translate_data_en_in = [{'input': src_text, 'output': tgt_text} for src_text, tgt_text in zip(src_en, tgt_in)]
            translate_data_in_ja = [{'input': src_text, 'output': tgt_text} for src_text, tgt_text in zip(src_in, tgt_ja)]
            
            self.save_json(translate_data_en_in, save_dir, f'{sce}_en_in.json')
            self.save_json(translate_data_in_ja, save_dir, f'{sce}_in_ja.json')

    def save_json(self, data, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _dataloader(self, batchsize=32, aug=False, aug_version=None):
        """_summary_

        Args:
            batchsize (int, optional): batchsize. Defaults to 32.
            aug (bool, optional): Only for en-in translation. Use the augmented dataset if set to be True. Defaults to False.
            aug_version (str, optional): Version choose from [qwen7B, qwen32B] aug dataset. Defaults to None.
`   
        Returns:
            _type_: DataLoader (train_dataloader, test_dataloader, val_dataloader)
        """

        scenario = ['train', 'test', 'valid']           
        data = [pd.read_csv(self.base_path + s + ".csv") for s in scenario]
        train_data, test_data, val_data = data

        train_src, train_tgt = train_data[self.src].tolist(), train_data[self.tgt].tolist()       
        test_src, test_tgt = test_data[self.src].tolist(), test_data[self.tgt].tolist()
        val_src, val_tgt = val_data[self.src].tolist(), val_data[self.tgt].tolist()

        if aug:
            aug_path = f"./aug_data/aug_{aug_version}.json"
            aug_dataset = self._load_json_file(aug_path)
            aug_train_src, aug_train_tgt = [item['english_sentence'] for item in aug_dataset], [item['indonesian_sentence'] for item in aug_dataset]
            train_src, train_tgt = train_src.extend(aug_train_src), train_src.extend(aug_train_tgt)
           
        # Tokenize
        train_dataset = TranslationDataset(train_src, train_tgt, self.tokenizer, self.add_prompt)
        test_dataset = TranslationDataset(test_src, test_tgt, self.tokenizer, self.add_prompt)
        val_dataset = TranslationDataset(val_src, val_tgt, self.tokenizer, self.add_prompt)
        
        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        
        return train_dataloader, test_dataloader, val_dataloader
    

class LLM_inference():
    def __init__(self):
        self.__api_key = "sk-ocwzvtwfxierjaldthzznqinzclqkediqwaydpcqocdhsoki"
        self.base_url = "https://api.siliconflow.cn/v1"

    def api_inference(self, model_name, system_prompt, user_prompt, temperature=0):

        api_key = self.__api_key
        base_url = self.base_url

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model = model_name,
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user', 
                    'content': user_prompt
                }
            ],
            stream=True,
            temperature=temperature,
            max_tokens=256
        )

        total_content = ""

        for chunk in response:
            total_content+=chunk.choices[0].delta.content

        return total_content


class Model_inference():  
    def __init__(self):
        pass

    def model_inference(model, tokenizer, src_texts, device, max_length=128):
        """_summary_

        Args:
            model (class): _desrciption_
            tokenizer (class): _desrciption_
            src_texts (_type_): _desrciption_
            device (str): cuda or cpu
            max_length (int, optional): Max Inference Seq Length. Defaults to 128.

        Returns:
            _List[str]_: Inference texts
        """
        model.eval()
        translations = []

        for src_text in src_texts:

            input_ids = tokenizer(src_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length').input_ids.to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)

        return translations
    