import numpy as np
from lib import misc
from omegaconf import OmegaConf, MISSING, OmegaConf
from typing import List, Any
import os


def _hparams(defense, attack, target):
    """Global registry of hyperparams."""

    hparams = {}

    def _hparam(name, default_val):
        """Define a hyperparameter."""

        assert(name not in hparams)
        hparams[name] = default_val

    # Target specific 
    # if attack == 'AutoDAN':
        # _hparam('target_max_num_tokens', 2048) # AutoDAN needs large max_num_tokens
    # else:
    if attack == 'AlpacaEval':
        # alpacaeval requires even longer output
        _hparam('target_max_num_tokens', 3072) 
    elif attack == 'InstructionFollowing': 
        # instructionfollow requires longer output
        _hparam('target_max_num_tokens', 1024) 
    elif attack in ['PiQA', 'OpenBookQA']:
        _hparam('target_max_num_tokens', 20) 
    else:
        _hparam('target_max_num_tokens', 200)
    _hparam('target_temperature', 0.0)
    _hparam('target_top_p', 1.0)

    # Attack specific
    if attack == 'GCG':
        _hparam('gcg_max_num_tokens', 100)

    # Defense specific
    # TARGET_MODEL_NAME = 'vicuna-vllm'
    # TARGET_MODEL_NAME = 'llama-2-vllm'
    # TARGET_MODEL_NAME = 'mistral-vllm'
    # TARGET_MODEL_NAME = 'gpt-3.5-turbo-0613'
    TARGET_MODEL_NAME = target
    name = TARGET_MODEL_NAME.split("-vllm")[0]
    _hparam('template_dir', os.environ.get("PROMPT_TAMPLATE_DIR", f"data/prompt-template-{name}"))
    _hparam('VIRTUAL_LLM_URL', os.environ.get('VIRTUAL_LLM_URL', 'http://localhost:10000/v1/'))

    if 'llama' in target:
        _hparam('base_filter', True)

    # SmoothLLM
    if 'SmoothLLM' in defense:
        _hparam('smoothllm_num_copies', 10)
        # _hparam('smoothllm_num_copies', 1)
        _hparam('judge_class', 'StringMatchJudge')
        _hparam('perturbation_temperature', 0.5)
        # _hparam('perturbation_temperature', 0.7)
        _hparam('perturbation_top_p', 0.5)
        if attack == 'AutoDANTransfer':
            # AutoDAN prompts are very long, we expand the max num tokens
            _hparam('perturbation_max_num_tokens', 2048) 
        else:
            _hparam('perturbation_max_num_tokens', 512)
        
        if defense == 'SmoothLLM':
            _hparam('smoothllm_pert_pct', 10)
            _hparam('smoothllm_perturbations', [
                'RandomSwap'
            ])

        elif defense == 'SemanticSmoothLLM':
            _hparam('smoothllm_perturbations', [
                'RandomSwap', 
            ])
            _hparam('smoothllm_pert_pct', 10)
            _hparam('perturbation_llm', TARGET_MODEL_NAME)
            _hparam('judge_llm', 'gpt-3.5-turbo-1106')
            _hparam('judge_temperature', 0.0)
            _hparam('judge_max_num_tokens', 100)
            _hparam('judge_top_p', 1.0)

    #  Baseline paraphrase defense
    elif defense == 'ParaphraseDefense':
        _hparam('perturbation_temperature', 0.7)
        _hparam('perturbation_top_p', 1.0)
        if attack == 'AutoDAN':
            _hparam('perturbation_max_num_tokens', 2048)
        else:
            _hparam('perturbation_max_num_tokens', 512)
        _hparam('perturbation_llm', 'gpt-3.5-turbo-1106')
    
    elif defense == 'InContextDefense':
        _hparam('incontext_nshot', 2)
    
    elif defense == 'LLMFilter' or defense == 'EraseAndCheck':
        _hparam('filter_llm', TARGET_MODEL_NAME)
        _hparam('filter_max_num_tokens', 100)
        _hparam('filter_temperature', 0.0)
        _hparam('filter_top_p', 1.0)

        if defense == 'EraseAndCheck':
            _hparam('erase_and_check_mode', 'suffix')
            _hparam('erase_and_check_max_erase', 20)
            _hparam('erase_and_check_randomized', False)
            _hparam('erase_and_check_prompt_sampling_ratio', 0.2)

    return hparams

def default_hparams(defense, attack, target):
    return _hparams(defense, attack, target)

def merge_cmd_configs(args, hparams):
    """ Merge command line configs with hparams. Override any config in hparams. """
    cmd_configs = OmegaConf.from_dotlist(args)
    # Merge command line configs with hparams
    for key, val in cmd_configs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise Warning(f"Unknown cmd config: {key}, will be ignored.")
    
    return hparams 
