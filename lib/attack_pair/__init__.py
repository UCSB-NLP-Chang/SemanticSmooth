# This folder contains the same code as the original PAIRAttack repo
# minimalist modification to make it work in our codebase

import torch

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from .language_models import GPT, HuggingFace
import lib.attack_pair.common as common
from .system_prompts import *
from .judges import load_judge
from omegaconf import OmegaConf

def get_model_path_and_template(model_name):
    BASEDIR = os.getenv("BASEDIR")
    def remap_conf(conf):
        conf['BASEDIR'] = conf
        return BASEDIR
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo-1106",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path": remap_conf(OmegaConf.load("config/llm/vicuna.yaml"))['model_path'],
            "template":"vicuna_v1.5"
        },
        "llama-2":{
            "path": remap_conf(OmegaConf.load("config/llm/llama-2.yaml"))['model_path'],
            "template":"llama-2"
        },
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
    

def load_indiv_model(model_name, device=None, hparams=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-1106"]:
        lm = GPT(model_name)
    else: 
        # always use vicuna
        lm = LocalVirtualLLM(
            model_name=model_path,
            model_template=get_conversation_template(template)
        )

    return lm, template

class PAIRAttackLLM:
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self, 
        model_name: str, 
        max_n_tokens: int, 
        max_n_attack_attempts: int, 
        temperature: float,
        top_p: float,
        conversation_num : int = 1, # the budget for attack
    ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.conversation_num = conversation_num
        self.keep_last_n = 2
        
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def reset_new_goal(self, goal, target_str):
        self.goal = goal
        self.target_str = target_str
        self.init_msg = common.get_init_msg(self.goal, self.target_str)
        self.convs_list = [common.conv_template(self.template) for _ in range(self.conversation_num)]
        self.system_prompt = get_attacker_system_prompt(
            self.goal,
            self.target_str
        )
        for conv in self.convs_list:
            conv.system_message = self.system_prompt

        self.iteration = 0
        self.judge_scores = []
        self.target_response_list = []
        self.processed_response_list = [self.init_msg for _ in range(self.conversation_num)]
        self.adv_prompt_list = []

        self.judge_lm = load_judge(OmegaConf.create({
            "judge_model" : 'gpt-3.5-turbo-1106',
            "judge_max_n_tokens" : 15,
            "judge_temperature" : 0,
            "goal" : goal,
            "target_str" : target_str,
        }))

    @torch.no_grad() 
    def generate_attack(self, prompt, target_response):
        if self.iteration != 0:
            self.target_response_list = [target_response]
            self.judge_scores = self.judge_lm.score(
                self.adv_prompt_list, self.target_response_list
            )
            for i, conv in enumerate(self.convs_list):
                for prev_n in range(-self.keep_last_n, -1):
                    token = self.model.tokenizer(
                        conv.system_message + "\n"+ " role: ".join([x[1] for x in conv.messages[2 * prev_n:]]))['input_ids']
                    if len(token) <= 3000:
                        conv.messages = conv.messages[2*prev_n: ]

        if self.iteration > 1:
            self.processed_response_list = [
                common.process_target_response(target_response, score, self.goal, self.target_str) 
                    for target_response, score in zip(self.target_response_list, self.judge_scores)]

        # Get adversarial prompts and improvement
        extracted_attack_list = self.get_attack(self.convs_list, self.processed_response_list)
        print("Finished getting adversarial prompts.")

        if all([attack is None for attack in extracted_attack_list]):
            return None

        # Extract prompts and improvements
        self.adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        self.improv_list = [attack["improvement"] for attack in extracted_attack_list]
        self.iteration += 1
        return self.adv_prompt_list[0]


    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            # elif 'vicuna' in self.model_name: # use local vllm model
            #     conv.append_message(conv.roles[1], init_message)
            #     full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            # outputs_list = self.model.batched_generate(full_prompts_subset,
            outputs_list = self.model(full_prompts_subset,
                    max_num_tokens = self.max_n_tokens,  
                    temperature = self.temperature,
                    top_p = self.top_p
            )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    if not full_output.strip().startswith("{\"improvement\": \""): 
                        full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output, self.model.stops)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

# Copied here from lib/language_models.py
import openai 
import concurrent
import time
from fastchat.model import get_conversation_template
import random
class APILanguageModel:

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 50
    CACHE_DIR = "cache"

    def __init__(self, model_config, hparams):
        super(APILanguageModel, self).__init__()
        self.model_config = model_config
        self.hparams = hparams

    def __call__(self, batch, temperature=0.0, max_num_tokens=100, top_p=1.0):
        
        def single_call(conv):
            output = ""
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.query_openai_api(
                        conv, temperature, max_num_tokens, top_p
                    )
                    # output = response["choices"][0]["message"]["content"]
                    output = response["choices"][0]["text"]
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
            return output
           
        # results = [single_call(conv) for conv in batch] 
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [executor.submit(single_call, conv) for conv in batch]
            concurrent.futures.wait(futures)
            results = [future.result() for future in futures]

        return results 
    
    def query_openai_api(self, conv, temperature, max_num_tokens, top_p, seed=None):
        # func = self.query_cacher(self.client.chat.completions.create)
        func = self.client.chat.completions.create
        
        if temperature != 0 and seed is None:
            seed = random.randint(0, 2**32-1)
        elif temperature == 0:
            seed = 42 if seed is None else seed

        # response = self.client.chat.completions.create(
        response = func(
            model=self.model_config['model_name'],
            messages=conv,
            max_tokens=max_num_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.API_TIMEOUT,
            response_format={ "type": "json_object" },
            seed=seed, 
            # use seed to make sure cache does not return same result for query with same message/temperature/top-p
        )
        if not isinstance(response, dict):
            response = response.dict()
        return response

VLLMURL = "http://localhost:10000/v1/"

class LocalVirtualLLM(APILanguageModel):
    """ Apply an API call to a local vllm server, return the API response.
        This is used when we have a local vLLM server
        See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    """

    def __init__(self, model_name, model_template):

        super(LocalVirtualLLM, self).__init__({}, {})        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stops = [
            "}"
        ] + [self.tokenizer.decode(x) for x in [29913, 9092, 16675]]
        self.client = openai.Client(
            api_key = "EMPTY",
            base_url=VLLMURL,
        )

        self.conv_template = model_template
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def query_openai_api(self, conv, temperature, max_num_tokens, top_p, seed=None):
        return self.client.completions.create(
            model=self.model_name,
            prompt=conv,
            max_tokens=max_num_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.API_TIMEOUT,
            stop=self.stops, 
        ).dict()