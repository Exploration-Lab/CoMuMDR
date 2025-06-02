from transformers import BertTokenizer, AutoTokenizer, AutoConfig

class BertTokenHelper(object):
    def __init__(self, bert_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir) # to use local file 
        special_tokens_dict = {'additional_special_tokens': ['URL', 'FILEPATH', '<root>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print("Load bert vocabulary finished.")


    def pad_token_id(self): 
        return self.tokenizer.pad_token_id


    # def batch_bert_id(self, inst_text):
    #     for idx, text in enumerate(inst_text):
    #         inst_text[idx] = text.replace('##', '@@')
    #     outputs = self.tokenizer.batch_encode_plus(inst_text, add_special_tokens=True)
    #     input_ids = outputs.data['input_ids']
    #     token_type_ids = outputs.data['token_type_ids']
    #     attention_mask = outputs.data['attention_mask']

    #     return input_ids, token_type_ids, attention_mask

    '''for xml-roberta'''
    def batch_bert_id(self, inst_text):
        for idx, text in enumerate(inst_text):
            inst_text[idx] = text.replace('##', '@@')
        
        # Tokenization with padding and truncation
        outputs = self.tokenizer.batch_encode_plus(
            inst_text, 
            add_special_tokens=True, 
            padding=True, 
            truncation=True,
            return_attention_mask=True, 
            return_token_type_ids=False  # Disable token_type_ids
        )

        input_ids = outputs['input_ids']
        attention_mask = outputs['attention_mask']
        
        # Generate token_type_ids manually as all zeros (standard for single-segment inputs)
        token_type_ids = [[0] * len(ids) for ids in input_ids]

        return input_ids, token_type_ids, attention_mask







