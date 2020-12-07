import argparse
import os
import sys
import pandas as pd
import traci
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from utc_reproduction import SumoGridEnvironment, Grid3x3Model


NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 6000


# Register the model and the environment
ModelCatalog.register_custom_model("grid_actor_critic", Grid3x3Model)
register_env(
    "3x3grid",
    lambda _: SumoGridEnvironment(
        net_file='nets/3x3/3x3.net.xml',
        route_folder='nets/3x3/routes/',
        num_cols=3,
        num_rows=3,
        num_train_steps=TRAIN_BATCH_SIZE * 1.0 / NUM_WORKERS,
        use_gui=False,
        num_seconds=14050,
        time_to_load_vehicles=2,
        max_depart_delay=0,
        out_csv_name="outputs/3x3grid/ppo",
    ),
)

ray.init()

trainer = PPOTrainer(
    env="3x3grid",
    config={
        "framework": "torch",
        "train_batch_size": TRAIN_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "sgd_minibatch_size": 256,
        "num_gpus": 0.5,
        "model": {
            "custom_model": "grid_actor_critic",
            "custom_model_config": {
                "num_tls": 3 * 3,
                "num_train_steps": TRAIN_BATCH_SIZE * 1.0 / NUM_WORKERS,
            },
        },
    }
)

result = trainer.train()
print(result)
