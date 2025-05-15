import os
from transformers import AutoModelForCausalLM
from open_clip import create_model_from_pretrained, get_tokenizer

class EmbeddingModelLoader:
    def __init__(self, model_name:str):
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.context_length = None

    def load_model_and_tokenizer(self):
        if self.model_name == "biomedclip":
            self.model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self.model, self.preprocess = create_model_from_pretrained(self.model_name)
            self.tokenizer = get_tokenizer(self.model_name)
            self.context_length = 256
            return self.model, self.preprocess, self.tokenizer, self.context_length
