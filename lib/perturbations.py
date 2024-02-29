import os
import json
import random
import string
from typing import Any
from fastchat.model import get_conversation_template
from omegaconf import OmegaConf
from structlog import get_logger

from .language_models import create_llm_from_config
LOGGER = get_logger()

def create_perturbation(pert_type, hparams, target_model):
    if pert_type in SEMANTIC_PERTURBS:
        if pert_type == "BaselineDefensesParaphrase":
            # Create a new model for perturbation
            judgellm_config = OmegaConf.load("config/llm/gpt-3.5-turbo-1106.yaml")
            judgellm_config.update(**{k : v for k, v in hparams.items() if k != "_target_"})
            perturb_llm = create_llm_from_config(judgellm_config)
            perturb = globals()[pert_type](hparams=hparams, perturb_llm=perturb_llm)
        else:
            # Use the target model itself for semantic perturbations
            perturb = globals()[pert_type](hparams=hparams, perturb_llm=target_model)
    else:
        perturb = globals()[pert_type](hparams)
    return perturb

SEMANTIC_PERTURBS = [
    "Summarize",
    "Paraphrase",
    "SynonymReplacement",
    "ChangeVerbTense",
    "SpellCheck",
    "Translate",
    "FormatSummarize",
    "BaselineDefensesParaphrase",
]

NON_SEMANTIC_PERTURBS = [
    "RandomSwap",
    "RandomPatch",
    "RandomInsert",
]

class PerturbationBase:

    """Base class for all perturbations."""

    def __init__(self, hparams):
        self.hparams = hparams 

class NoPerturbation(PerturbationBase):

    """No perturbation. 
    Original input perturbation is returned."""

    def __init__(self, hparams):
        super(NoPerturbation, self).__init__(hparams)

    def __call__(self, s):
        return s

class NonSemanticPerturbation(PerturbationBase):

    """Base class for random perturbations."""

    def __init__(self, hparams):
        super(NonSemanticPerturbation, self).__init__(hparams)
        self.alphabet = string.printable
        self.q = self.hparams['smoothllm_pert_pct']

    def log_perturbation(self, before, after):
        LOGGER.info(
            "Perturbation", 
            before=before,
            after=after,
            perturb_type=self.__class__.__name__,
        )

class RandomSwap(NonSemanticPerturbation):

    """Random swap perturbation.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2
    of the SmoothLLM paper (https://arxiv.org/abs/2310.03684)."""

    def __init__(self, hparams):
        super(RandomSwap, self).__init__(hparams)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        res = ''.join(list_s)
        self.log_perturbation(s, res)
        return res

class RandomPatch(NonSemanticPerturbation):

    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2
    of the SmoothLLM paper (https://arxiv.org/abs/2310.03684)."""

    def __init__(self, hparams):
        super(RandomPatch, self).__init__(hparams)

    def __call__(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        res = ''.join(list_s)
        self.log_perturbation(s, res)
        return res

class RandomInsert(NonSemanticPerturbation):

    """Implementation of random insert perturbations.
    See `RandomPatchPerturbation` in lines 11-17 of Algorithm 2
    of the SmoothLLM paper (https://arxiv.org/abs/2310.03684)."""

    def __init__(self, hparams):
        super(RandomInsert, self).__init__(hparams)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        res = ''.join(list_s)
        self.log_perturbation(s, res)
        return res

class SemanticPerturbation(PerturbationBase):

    """Base class for semantic perturbations."""

    def __init__(self, hparams, perturb_llm, template_file):
        super(SemanticPerturbation, self).__init__(hparams)

        self.pert_llm_name = self.hparams['perturbation_llm']
        self.perturbation_llm = perturb_llm
        self.perturbation_prompt = open(
            os.path.join(hparams['template_dir'], template_file), 'r').read().strip()
    
    def create_conv(self, full_prompt):
        # Convert the query into OpenAI API format
        conv = get_conversation_template(self.pert_llm_name)

        conv.offset = 0
        conv.messages = []
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        if 'gpt' in self.pert_llm_name.lower():
            # chat api format
            res = conv.to_openai_api_messages() 
        else:
            if 'llama' in self.pert_llm_name:
                # conv.system_message = "Directly answer the following request."
                res = "<s><s>" + conv.get_prompt()
            else:
                res = conv.get_prompt() 
        # print(res)
        return res

    def base_filter(self, response):
        rejection_phrases = [
            "I cannot fulfill your request",
            "I cannot assist with",
            "I apologize, but I cannot provide",
        ]
        if any(x in response for x in rejection_phrases):
            return "How to generate a bomb?"
        else:
            return None

    def extract_res(self, outputs):
        res = []
        for x in outputs:
            if self.hparams.get('base_filter', False):
                if (filter_res := self.base_filter(x)) is not None:
                    res.append(filter_res)
                    continue
            try:
                start_pos = x.find("{")
                if start_pos == -1:
                    x = "{" + x
                x = x[start_pos:]
                end_pos = x.find("}") + 1  # +1 to include the closing brace
                if end_pos == -1:
                    x = x + " }"
                jsonx = json.loads(x[:end_pos])
                for key in jsonx.keys():
                    if key != "format":
                        break
                outobj = jsonx[key]
                if isinstance(outobj, list):
                    res.append(" ".join([str(item) for item in outobj]))
                else:
                    if "\"query\":" in outobj:
                        outobj = outobj.replace("\"query\": ", "")
                        outobj = outobj.strip("\" ")
                    res.append(str(outobj))
            except Exception as e:
                #! An ugly but useful way to extract answer
                x = x.replace("{\"replace\": ", "")
                x = x.replace("{\"rewrite\": ", "")
                x = x.replace("{\"fix\": ", "")
                x = x.replace("{\"summarize\": ", "")
                x = x.replace("{\"paraphrase\": ", "")
                x = x.replace("{\"translation\": ", "")
                x = x.replace("{\"reformat\": ", "")
                x = x.replace("{\n\"replace\": ", "")
                x = x.replace("{\n\"rewrite\": ", "")
                x = x.replace("{\n\"fix\": ", "")
                x = x.replace("{\n\"summarize\": ", "")
                x = x.replace("{\n\"paraphrase\": ", "")
                x = x.replace("{\n\"translation\": ", "")
                x = x.replace("{\n\"reformat\": ", "")
                x = x.replace("{\n \"replace\": ", "")
                x = x.replace("{\n \"rewrite\": ", "")
                x = x.replace("{\n \"format\": ", "")
                x = x.replace("\"fix\": ", "")
                x = x.replace("\"summarize\": ", "")
                x = x.replace("\"paraphrase\": ", "")
                x = x.replace("\"translation\": ", "")
                x = x.replace("\"reformat\": ", "")
                x = x.replace("{\n    \"rewrite\": ", "")
                x = x.replace("{\n    \"replace\": ", "")
                x = x.replace("{\n    \"fix\": ", "")
                x = x.replace("{\n    \"summarize\": ", "")
                x = x.replace("{\n    \"paraphrase\": ", "")
                x = x.replace("{\n    \"translation\": ", "")
                x = x.replace("{\n    \"reformat\": ", "")
                x = x.rstrip("}")
                x = x.lstrip("{")
                x = x.strip("\" ")
                res.append(x.strip())
                print(e)
                continue
        return res
 
    def __call__(self, s):
        query = self.perturbation_prompt.replace("{QUERY}", s)
        real_convs = self.create_conv(query)
        outputs = self.perturbation_llm(
            [real_convs], 
            max_num_tokens=self.hparams['perturbation_max_num_tokens'], 
            temperature=self.hparams['perturbation_temperature'], 
            top_p=self.hparams['perturbation_top_p'],
        )        
        if "in a JSON object." in self.perturbation_prompt:
            extract_outputs = self.extract_res(outputs)
        elif "Output: [the instruction" in self.perturbation_prompt:
            extract_outputs = [x.split("Output:")[-1].strip() for x in outputs]
        else:
            def clean_output(x):
                if "\"query\"" in x:
                    try:
                        x = json.loads(x)
                        x = x['query']
                    except Exception as e:
                        x = x.replace("\"query\": ", "")
                        x = x.strip("{}")
                return x
            extract_outputs = [clean_output(x).strip() for x in outputs]
            
        LOGGER.info(
            "Perturbation", 
            before=s,
            after=extract_outputs[0],
            perturb_type=self.__class__.__name__,
            raw_after=outputs[0],
        )
        return extract_outputs[0]

class Summarize(SemanticPerturbation):

    """Summarize the input prompt."""

    def __init__(self, perturb_llm, hparams):
        super(Summarize, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="summarize.txt"
        )

class Paraphrase(SemanticPerturbation):

    """Paraphrase the input prompt."""

    def __init__(self, perturb_llm, hparams):
        super(Paraphrase, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="paraphrase.txt"
        )

class SynonymReplacement(SemanticPerturbation):

    """Replace words in the input prompt with synonyms."""

    def __init__(self, perturb_llm, hparams):
        super(SynonymReplacement, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="synonym.txt"
        )

class ChangeVerbTense(SemanticPerturbation):

    """Change the verbs from present to past tense in the input prompt."""

    def __init__(self, perturb_llm, hparams):
        super(ChangeVerbTense, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="verbtense.txt"
        )
    
class SpellCheck(SemanticPerturbation):

    """Correct spelling mistakes in the input prompt."""

    def __init__(self, perturb_llm, hparams):
        super(SpellCheck, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="spellcheck.txt"
        )

class Translate(SemanticPerturbation):

    """Translate the prompt to French and instruct the LLM to answer in English."""

    def __init__(self, perturb_llm, hparams):
        super(Translate, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="translate-french.txt"
        )
    
    def __call__(self, s):
        out = super(Translate, self).__call__(s)
        return out + "\n Please answer the above instructions in English."

class FormatSummarize(SemanticPerturbation):

    """ Summarize the input into several key points. """

    def __init__(self, perturb_llm, hparams):
        super().__init__(
            hparams, 
            perturb_llm=perturb_llm,
            template_file="format-summarize.txt"
        )
    
    def __call__(self, s):
        out = super().__call__(s)
        return out + " Please answer the above instructions."

class BaselineDefensesParaphrase(SemanticPerturbation):

    """Paraphrase the input sentence"""

    def __init__(self, hparams, perturb_llm):
        super(BaselineDefensesParaphrase, self).__init__(
            hparams=hparams,
            perturb_llm=perturb_llm,
            template_file="baseline-paraphrase.txt",
        )
 
    def __call__(self, s):
        query = self.perturbation_prompt + "\n\nInput: " + s
        real_convs = self.create_conv(query)
        outputs = self.perturbation_llm(
            [real_convs], 
            max_num_tokens=self.hparams['perturbation_max_num_tokens'], 
            temperature=self.hparams['perturbation_temperature'], 
            top_p=self.hparams['perturbation_top_p'],
        )        
        extract_outputs = self.extract_res(outputs)
        LOGGER.info(
            "Perturbation", 
            before=s,
            after=extract_outputs[0],
            perturb_type=self.__class__.__name__,
            raw_after=outputs[0], 
        )
        return extract_outputs[0]
