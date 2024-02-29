from typing import List, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class DefenseConfig:
    pass

@dataclass
class SmoothDefenseConfig(DefenseConfig):
    perturbation_max_num_tokens: int
    smoothllm_num_copies : int
    judge_class: str
    perturbation_temperature: float
    perturbation_top_p: float

@dataclass
class SmoothLLMConfig(SmoothDefenseConfig):
    smoothllm_pert_pct: int
    smoothllm_perturbations: List[str]

@dataclass
class SemanticSmoothConfig(SmoothDefenseConfig):
    perturbation_llm : str
    judge_llm: str
    smoothllm_perturbations: List[str]
    judge_temperature: float
    judge_max_num_tokens: int
    judge_top_p: float
    policy_type: str
    policy_path: str

@dataclass
class ParaphraseDefenseConfig(DefenseConfig):
    perturbation_num_tokens: int
    perturbation_temperature: float
    perturbation_top_p: float
    perturbation_llm: str

@dataclass
class InContextDefense(DefenseConfig):
    incontext_nshot: int

@dataclass
class LLMFilterDefenseConfig(DefenseConfig):
    filter_llm: str
    filter_max_num_tokens: float
    filter_temperature: float

@dataclass
class EraseAndCheckDefenseConfig(LLMFilterDefenseConfig):
    erase_and_check_mode: str
    erase_and_check_max_erase: int
    erase_and_check_randomized: bool
    erase_and_check_prompt_sampling_ratio: float
   
@dataclass
class TaskConfig:
    target_max_num_tokens: int
    target_temperature: float
    target_top_p: float
    attack_log_file: str
    
@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    tokenizer_path: str
    conversation_template: str

    target_max_num_tokens: int
    target_temperature: float
    target_top_p: float

@dataclass
class AttackConfig:
    attack_type: str
    max_iter: int
    attack_llm: str
    attack_max_num_tokens: int
    attack_max_n_attack_attempts: int
    

@dataclass
class ExperimentConfig:
    defense: DefenseConfig
    task: TaskConfig
    llm: ModelConfig
    VIRTUAL_LLM_URL: str
    BASEDIR: str
    OPENAI_API_KEY: str
