import json
import torch

from fastchat.model import get_conversation_template

class Prompt:

    """Wrapper for an input prompt to an LLM."""

    def __init__(self, full_prompt, perturbable_prompt, goal, target):
        """
        Parameters:
            full_prompt: 
        """

        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.goal = goal
        self.target = target

        # identify whether the prompt is perturbed, will be set in perturbation
        self.perturb_type = ""

    def perturb(self, perturbation_fn):
        """ Perturb the current prompt using perturbation_fn
        Args:
            perturbation_fn: a function that applies perturbation
        """

        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        try:
            self.full_prompt = self.full_prompt.replace(
                self.perturbable_prompt,
                perturbed_prompt
            )
        except Exception as e:
            print(e)
        self.perturbable_prompt = perturbed_prompt  
        self.perturb_type = perturbation_fn.__class__.__name__
    
    def construct_conv(self, conv_template):
        conv = get_conversation_template(conv_template)
        conv.offset = 0
        conv.messages = []
        if "llama" in conv_template.lower():
            conv.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information"
        conv.append_message(conv.roles[0], self.perturbable_prompt)
        conv.append_message(conv.roles[1], "")
        return conv
    
    def to_openai_api_messages(self, conv_template):
        conv = self.construct_conv(conv_template)
        res = conv.to_openai_api_messages()
        return res
    
    def new_full_prompt(self, conv_template):
        conv = self.construct_conv(conv_template)
        full_prompt = conv.get_prompt()
        return full_prompt
    
    def set_slices(self, tokenizer, conv_template, prefix=None, target=None):

        conv = get_conversation_template(conv_template)
        conv.offset = 0
        conv.messages = []
        if prefix is None:
            prefix = self.perturbable_prompt
        if target is None:
            target = self.target
        
        self.perturbable_prompt = prefix
        self.target = target

        conv.append_message(conv.roles[0], f"{prefix}")
        conv.append_message(conv.roles[1], f"{target}")
        full_prompt = conv.get_prompt()

        encoding = tokenizer(full_prompt)
        toks = encoding.input_ids
        
        if conv_template == 'llama-2':
            conv = get_conversation_template(conv_template)
            conv.messages = []

            conv.append_message(conv.roles[0], None)
            toks = tokenizer(conv.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            conv.update_last_message(f"{prefix}")
            toks = tokenizer(conv.get_prompt()).input_ids
            self._prefix_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            conv.append_message(conv.roles[1], None)
            toks = tokenizer(conv.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._prefix_slice.stop, len(toks))

            conv.update_last_message(f"{target}")
            toks = tokenizer(conv.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)
        else:
            python_tokenizer = False or conv_template == 'oasst_pythia'
            try:
                encoding.char_to_token(len(full_prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                conv = get_conversation_template(conv_template)
                conv.offset = 0
                conv.messages = []

                conv.append_message(conv.roles[0], None)
                toks = tokenizer(conv.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                conv.update_last_message(f"{prefix}")
                toks = tokenizer(conv.get_prompt()).input_ids
                self._prefix_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))

                conv.append_message(conv.roles[1], None)
                toks = tokenizer(conv.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._prefix_slice.stop, len(toks))

                conv.update_last_message(f"{target}")
                toks = tokenizer(conv.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(conv.system_message))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(full_prompt.find(conv.roles[0])),
                    encoding.char_to_token(
                        full_prompt.find(conv.roles[0]) + len(conv.roles[0]) + 1)
                )
                self._prefix_slice = slice(
                    encoding.char_to_token(full_prompt.find(prefix)),
                    encoding.char_to_token(full_prompt.find(prefix) + len(prefix))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(full_prompt.find(conv.roles[1])),
                    encoding.char_to_token(
                        full_prompt.find(conv.roles[1]) + len(conv.roles[1]) + 1)
                )
                print(full_prompt.find(target))
                self._target_slice = slice(
                    encoding.char_to_token(full_prompt.find(target)),
                    encoding.char_to_token(full_prompt.find(target) + len(target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(full_prompt.find(target)) - 1,
                    encoding.char_to_token(full_prompt.find(target) + len(target)) - 1
                )
        
        return full_prompt

    def get_input_ids(self, tokenizer, conv_template, target=None):
        prompt = self.set_slices(tokenizer, conv_template, target=target)
        toks = tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        return input_ids

   
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return f"""Prompt instance.

        Full prompt: {self.full_prompt}

        Perturbable prompt: {self.perturbable_prompt}
        
        Goal: {self.goal}
        """
        return 