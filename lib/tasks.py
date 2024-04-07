import json
import pandas as pd
import os
import copy

from lib import prompts
from datasets import load_dataset
from fastchat.model import get_conversation_template

def create_task_from_config(task_config, target_llm):
    task_class = globals()[task_config._target_]
    return task_class(target_llm)

ATTACKS = [
    "GCG",
    "PAIR",
    "AutoDANTransfer",
]

BENIGN_TASKS = [
    "InstructionFollowing",
    "AlpacaEval",
]

ALL_ATTACKS = ATTACKS + BENIGN_TASKS

class Tasks:

    """Base class for jailbreaking attacks."""

    def __init__(self, target_model):
        self.target_model = target_model

    def create_prompt(self, jailbreak_str, goal, target):
        """Create a fastchat conversation template, where
        the `user` role is a jailbreaking string and
        the `assistant` role is empty.
        
        Parameters:
            jailbreak_str: Jailbreaking string for LLM.
            goal: Goal string for the jailbreak.

        Returns:
            full_prompt: Concatenated prompt with `user` and
                `assistant` roles.
        """

        conv_template = get_conversation_template(
            self.target_model.conv_template.name
        )

        # Append jailbreaking string to conversation
        conv_template.append_message(
            role=conv_template.roles[0], 
            message=jailbreak_str
        )

        # Append blank response string to conversation
        conv_template.append_message(
            role=conv_template.roles[1],
            message=""
        )

        # Extract the full prompt and clear the conversation
        if conv_template.name in ['chatgpt', 'gpt-3.5-turbo']:
            # NOTE: No real prompt for gpt models
            full_prompt = "\n".join([x['role'] + ": " + x['content'] for x in conv_template.to_openai_api_messages()])
        else:
            full_prompt = conv_template.get_prompt()
        conv_template.messages = []

        return prompts.Prompt(
            full_prompt=full_prompt,
            perturbable_prompt=jailbreak_str,
            goal=goal,
            target=target,
        )

class GCG(Tasks):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, target_model, logfile='data/GCG/vicuna.json'):
        super(GCG, self).__init__(target_model)

        with open(logfile, 'r') as f:
            log = json.load(f)

        goals, controls, targets = log['goal'], log['controls'], log['target']

        self.prompts = [
            self.create_prompt(f"{goal} {control}", goal, target)
            for (goal, control, target) in zip(goals, controls, targets)
        ]

class PAIR(Tasks):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, target_model, logfile='data/PAIR/vicuna_jailbreaks.pd'):
        super(PAIR, self).__init__(target_model)

        df = pd.read_pickle(logfile)
        self.prompts = []
        for i in range(len(df)):
            obj = df.iloc[i].to_dict()
            self.prompts.append(
                self.create_prompt(
                    obj['jailbreak_prompt'],
                    obj['goal'],
                    obj['target_str'],
                )
            )

class AutoDANTransfer(Tasks):

    """ AutoDANTransfer attack.

    Title: AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models
    Authors: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao
    Paper: https://arxiv.org/abs/2310.04451
    Code: https://github.com/sheltonliu-n/autodan
    """

    def __init__(self, target_model, logfile='data/AutoDAN/vicuna.json'):
        super(AutoDANTransfer, self).__init__(target_model)
        
        raw = json.load(open(logfile))        
        self.prompts = []
        for i in range(len(raw['goal'])):
            self.prompts.append(
                self.create_prompt(
                    raw['jailbreak_prompt'][i] + " " + raw['goal'][i],
                    raw['goal'][i],
                    raw['target'][i],
                )
            )

class StrongReject(Tasks):


    """ StrongReject dataset.

    Title: A StrongREJECT for Empty Jailbreaks
    Authors: Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, Sam Toyer
    Paper: https://arxiv.org/abs/2402.10260
    Code&data: https://github.com/alexandrasouly/strongreject
    """

    def __init__(self, target_model, hparams, logfile='data/strongreject_dataset/strongreject_small_dataset.csv'):
        super().__init__(target_model, hparams)

        self.prompts = []
        df = pd.read_csv(logfile)
        for idx, line in df.iterrows():
            line = line.to_dict()
            self.prompts.append(
                self.create_prompt(line['forbidden_prompt'], "", "")
            )
            
class InstructionFollowing(Tasks):

    """ Benign instructions.

    Title: Instruction-Following Evaluation for Large Language Models
    Authors: Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou
    Paper: https://arxiv.org/abs/2311.07911
    Code&data: https://github.com/google-research/google-research/tree/master/instruction_following_eval 
    """

    def __init__(self, target_model, logfile='data/google-instruction-following/instfollow-125.jsonl'):
        super(InstructionFollowing, self).__init__(target_model)

        self.prompts = []
        for idx, line in enumerate(open(logfile, 'r').readlines()):
            line = json.loads(line.strip())
            self.prompts.append(
                self.create_prompt(line['prompt'], line['prompt'], "")
            )


class AlpacaEval(Tasks):

    """ AlpacaEval instructions.

    Title: AlpacaEval: An Automatic Evaluator of Instruction-following Models
    Authors: Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto
    Code&data: https://github.com/tatsu-lab/alpaca_eval
    """

    def __init__(self, target_model, logfile="data/AlpacaEval/AlpacaEval.jsonl"):
        super(AlpacaEval, self).__init__(target_model)

        self.prompts = []

        for idx, line in enumerate(open(logfile, 'r').readlines()):
            line = json.loads(line.strip())
            self.prompts.append(
                self.create_prompt(
                    line['instruction'], # the query
                    line['instruction'], # the query
                    line['output'], # the default output from text-davinci-003 
                )
            )


class PiQA(Tasks):
    def __init__(self, target_model, logfile=""):
        super(PiQA, self).__init__(target_model)

        self.prompts = []
        QATemplate = (
            "Question: {question}\n"
            "Choice A: {choice1}\n"
            "Choice B: {choice2}\n"
            "Which of the above is the correct answer? You answer should be one of the following Choice A, Choice B. No explanations"
        )
        
        data = load_dataset('piqa')['validation']
        data = data.select(range(500)) # use the first 500

        for idx, sample in enumerate(data):
            query = QATemplate.format(**{
                "question" : sample['goal'],
                "choice1": sample['sol1'],
                "choice2": sample['sol2'],
            })
            self.prompts.append(
                self.create_prompt(
                    query, # The useful one
                    sample['goal'] + sample['sol1'] + sample['sol2'], #! will be used as the key for answer
                    sample['label']
                )
            )

        
class OpenBookQA(Tasks):
    def __init__(self, target_model, logfile=""):
        super(OpenBookQA, self).__init__(target_model)

        self.prompts = []

        QATemplate = (
            "Question: {question}\n"
            "Choice A: {choice1}\n"
            "Choice B: {choice2}\n"
            "Choice C: {choice3}\n"
            "Choice D: {choice4}\n"
            "Which of the above is the correct answer? You answer shoud be one of the following Choice A, Choice B, Choice C, Choice D. No explanations."
        )

        data = load_dataset('openbookqa')['test']
        for idx, sample in enumerate(data):
            query = QATemplate.format(**{
                "question": sample['question_stem'],
                "choice1": sample['choices']['text'][0],
                "choice2": sample['choices']['text'][1],
                "choice3": sample['choices']['text'][2],
                "choice4": sample['choices']['text'][3],
            })

            self.prompts.append(
                self.create_prompt(
                    query,
                    sample['question_stem'] + sample['choices']['text'][0],
                    sample['answerKey'],
                )
            )