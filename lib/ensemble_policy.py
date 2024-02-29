import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def create_ensemble_policy_model(hparams):
    return SentenceBERT(
        len(hparams['smoothllm_perturbations']), 
    )

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceBERT(nn.Module):
    def __init__(self, num_perturbations):
        super(SentenceBERT, self).__init__() 
        path = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.truncation_side = 'left'
        self.model = AutoModel.from_pretrained(path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_perturbations).to(self.model.device)

    def optimize_parameters(self):
        return self.classifier.parameters()
    
    def optimize_state_dict(self):
        return self.state_dict()
    
    def forward(self, input_str):
        inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')
        inputs = {k : v.to(self.model.device) for k, v in inputs.items()}
        encodings = self.model(**inputs)
        sentence_embeddings = mean_pooling(encodings, inputs['attention_mask'])
        logits = self.classifier(sentence_embeddings)
        return F.softmax(logits, dim=-1)
