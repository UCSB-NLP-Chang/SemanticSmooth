import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import openai
from typing import Any, List, Dict
import time
import concurrent.futures
import random


def create_llm_from_config(model_config):
    llm_name = model_config._target_
    if 'gpt' in llm_name:
        llm = GPT(model_config=model_config)
    elif llm_name.endswith('-vllm'):
        # NOTE: the model name ends with -vllm
        llm = VirtualLLM(model_config=model_config)
    else: 
        llm = HuggingFaceLLM(
            model_config=model_config,
            device='cuda:0'
        )
    return llm

class LanguageModelBase:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model_name = model_config['model_name']

class HuggingFaceLLM(LanguageModelBase):

    """Forward pass through a vanilla HuggingFace LLM."""

    def __init__(self, model_config, device='cuda:0'):

        super(HuggingFaceLLM, self).__init__(model_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['tokenizer_path'],
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in self.model_config['tokenizer_path']:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.conv_template = get_conversation_template(
            self.model_config['conversation_template']
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_path'],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        ).to(device).eval()
        
    def __call__(self, batch, temperature=0.0, max_num_tokens=100, top_p=1.0):

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        try:
            if temperature == 0.0:
                outputs = self.model.generate(
                    batch_input_ids, attention_mask=batch_attention_mask,
                    max_new_tokens=max_num_tokens,
                )
            else:
                outputs = self.model.generate(
                    batch_input_ids, 
                    attention_mask=batch_attention_mask, 
                    max_new_tokens=max_num_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )

        except RuntimeError:
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs

class APILanguageModel(LanguageModelBase):

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 90
    CACHE_DIR = "cache"

    def __init__(self, model_config):
        super(APILanguageModel, self).__init__(model_config)

    def __call__(self, batch, temperature=0.0, max_num_tokens=100, top_p=1.0, json_format=False):
        
        def single_call(conv):
            output = ""
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.query_openai_api(
                        conv, temperature, max_num_tokens, top_p, json_format=json_format
                    )
                    if 'message' in response['choices'][0]:
                        output = response["choices"][0]["message"]["content"]
                    else:
                        output = response["choices"][0]["text"]
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
            return output
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [executor.submit(single_call, conv) for conv in batch]
            concurrent.futures.wait(futures)
            results = [future.result() for future in futures]
            
        return results

    def query_openai_api(self, conv, temperature, max_num_tokens, top_p, seed=None, json_format=False):
        if temperature != 0 and seed is None:
            seed = random.randint(0, 2**32-1)
        elif temperature == 0:
            seed = 42 if seed is None else seed
        
        if json_format or (self.model_config['model_name'] in ['gpt-3.5-turbo-1106']): 
            other_arg =  {"response_format" : { "type": "json_object" }}
        else:
            other_arg =  {}
            
        response = self.client.chat.completions.create(
            model=self.model_config['model_name'],
            messages=conv,
            max_tokens=max_num_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.API_TIMEOUT,
            seed=seed, 
            **other_arg
        )
        if not isinstance(response, dict):
            response = response.dict()
        return response


class GPT(APILanguageModel):

    """
    Borrowed from https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py#L21
    """

    def __init__(self, model_config):

        super(GPT, self).__init__(model_config)
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
        assert OPENAI_API_KEY is not None, "Please set the OPENAI_API_KEY environment variable"
        self.client = openai.Client(api_key=OPENAI_API_KEY)
        self.conv_template = get_conversation_template("gpt-3.5-turbo")
        self.conv_template.name = "gpt-3.5-turbo"
        self.tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

class VirtualLLM(APILanguageModel):
    """ 
    API call to a local vllm server, return the API response.
    This is used when we have a local vLLM server
    See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    """

    def __init__(self, model_config):

        super(VirtualLLM, self).__init__(model_config)        
        self.client = openai.Client(
            api_key = "EMPTY", base_url=model_config.VIRTUAL_LLM_URL,
        )

        # TODO check whether it's able to merge with GPT()
        self.conv_template = get_conversation_template(
            self.model_config['conversation_template']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['tokenizer_path'],
            trust_remote_code=True,
            use_fast=False
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def query_openai_api(self, conv, temperature, max_num_tokens, top_p, seed=None, json_format=False, stops=None):
        #! Use continuation api for VirtualLLM model instead of chat API
        return self.client.completions.create(
            model=self.model_config['model_path'],
            prompt=conv,
            max_tokens=max_num_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.API_TIMEOUT,
            stop=stops, 
        ).dict()
