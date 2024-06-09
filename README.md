# Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing

This is the official implementation for the paper [Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing](https://arxiv.org/abs/2402.16192). 

## Environment Setup

1. Install conda environment with `environment.yaml`. 

2. Download the pre-trained LLMs into the `model_weights` directory.

3. Specify the path to model weights in `config/${MODELNAME}.yaml`. An example config is in `config/llm/vicuna.yaml`.

Our implementation also supports calling a local vllm LLM server to improve the generation speed, checkout `language_models.py:VirtualLLM` and `config/llm/vicuna-vllm.yaml` for more details.

## Usage

For transfer attack experiment:

```bash
python transfer_attack.py llm=${LLM} task=${TASK} defense=${ATTACK}
```

For adaptive attack experiment:
```bash
python adaptive_attack.py llm=${LLM} attacker=${ATTACK} defense=${DEFENSE} task=advbench
```

For training the dynamic selection policy:
```bash
python train_selector.py llm=${LLM}
```

Here, `LLM` specifies the target LLM to be applied. The corresponding config file in `config/llm` folder is loaded. Similar for `TASK`, `ATTACK`, and `DEFENSE`.

Complete configs are in the `config` directory with detailed comments. Please checkout there.

## Citation
If you find this work useful, please cite the following paper:
```bibtex
@article{ji2024defending,
  title   = {Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing},
  author  = {Jiabao Ji and Bairu Hou and Alexander Robey and George J. Pappas and Hamed Hassani and Yang Zhang and Eric Wong and Shiyu Chang},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2402.16192}
}

@article{robey2023smoothllm,
  title={SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks},
  author={Robey, Alexander and Wong, Eric and Hassani, Hamed and Pappas, George J},
  journal={arXiv preprint arXiv:2310.03684},
  year={2023}
}
```

Huge thanks to the following repos that greatly help our implementation: 
* [https://github.com/arobey1/smooth-llm](https://github.com/arobey1/smooth-llm)
* [https://github.com/SheltonLiu-N/AutoDAN](https://github.com/SheltonLiu-N/AutoDAN)
* [https://github.com/google-research/google-research/tree/master/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
