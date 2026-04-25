import wandb

from src.generated import ConfigModel
from src.utils.misc import generate_run_name


class Trainer:
    def __init__(self, cfg: ConfigModel) -> None:
        self.cfg = cfg

    def train(self) -> None:
        wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=generate_run_name(self.cfg.wandb_run_name),
            config=self.cfg.model_dump(),
        )

        try:
            pass
        finally:
            wandb.finish()

    def eval(self) -> None:
        pass
