import os
import hydra
import structlog
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import rich
from rich.progress import Progress
from tqdm import tqdm
from structlog.contextvars import bound_contextvars

from lib.config_util import ExperimentConfig
from lib.log import configure_structlog
from lib.misc import set_seed, seed_hash
from lib.tasks import create_task_from_config
from lib.language_models import create_llm_from_config
from lib.judges import create_eval_judge
from lib.defenses import create_defense_from_config

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(hparams: ExperimentConfig):
    OmegaConf.resolve(hparams)
    os.environ['OPENAI_API_KEY'] = hparams.OPENAI_API_KEY
    os.environ['BASEDIR'] = hparams.BASEDIR
    os.environ['VIRTUAL_LLM_URL'] = hparams.VIRTUAL_LLM_URL
    hydraconf = HydraConfig.get()

    # Set the gloal random seed
    if hparams.seed == -1:
        set_seed(seed_hash(*hparams))
    else:
        set_seed(hparams.seed)

    target_llm = create_llm_from_config(hparams.llm) 
    task = create_task_from_config(hparams.task, target_llm)
    # Merge generation config to defense
    full_defense_hparams = OmegaConf.create(hparams.defense)
    full_defense_hparams.update(**{k : v for k, v in hparams.task.items() if k != "_target_"})
    defense = create_defense_from_config(full_defense_hparams, target_llm)
    eval_judge = create_eval_judge(hparams.task._target_)

    # Main logic
    configure_structlog(f"{hydraconf.runtime.output_dir}/{hydraconf.job.name}.log")
    LOGGER = structlog.getLogger()
    
    # Add some colorful progress bar
    progressbar = Progress(
        rich.progress.TextColumn("[progress.description]{task.description} {task.completed}/{task.total}"), 
        rich.progress.BarColumn(), rich.progress.TaskProgressColumn(), rich.progress.TimeRemainingColumn(),
        disable=os.environ.get("POOR", False)
    )
    task_progress = progressbar.add_task("[blue]TaskProgress", total=len(task.prompts))
    
    all_queries = [] 
    all_goals = []
    all_outputs = []
    with progressbar:
        for i, prompt in enumerate(tqdm(task.prompts)):
            progressbar.update(task_progress, completed=i)
            with bound_contextvars(
                model=hparams.llm._target_, task=hparams.task._target_, index=i,
            ):
                output = defense(prompt)
                LOGGER.info(
                    "FinalModelOutput", 
                    attack_prompt=prompt.full_prompt,
                    goal=prompt.goal,
                    output=output,
                )

                all_goals.append(prompt.goal)
                all_queries.append(prompt.full_prompt)
                all_outputs.append(output)
                break

    judge_res = eval_judge(all_goals, all_queries, all_outputs)
    LOGGER.info("Task result", judgeresult=judge_res)
     
if __name__ == "__main__":
    main() 