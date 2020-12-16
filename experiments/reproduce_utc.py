import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from utc_reproduction import SumoGridEnvironment, Grid3x3Model
from datetime import datetime


NUM_WORKERS = 16
NUM_EPISODES = 50
ROLLOUT = 128
TRAIN_BATCH_SIZE = 3600 * NUM_WORKERS  # 4C

timestamp = str(datetime.now())


# Register the model and the environment
ModelCatalog.register_custom_model("grid_actor_critic", Grid3x3Model)
curr_idx = 0
register_env(
    "3x3grid",
    lambda config: SumoGridEnvironment(
        net_file='nets/3x3/3x3.net.xml',
        route_folder='nets/3x3/',
        additionals_file='nets/3x3/3x3.add.xml',
        num_cols=3,
        num_rows=3,
        num_train_steps=(TRAIN_BATCH_SIZE + ROLLOUT) * 1.0 / NUM_WORKERS,
        use_gui=False,
        num_seconds=3600,
        delta_time=1,
        time_to_load_vehicles=2,
        max_depart_delay=0,
        out_csv_name=f"outputs/reproduce/3x3grid/global_only/reproduce_timestamp_{timestamp}_train_",
        global_only=True,
    ),
)

ray.init()

trainer = PPOTrainer(
    env="3x3grid",
    config={
        "framework": "torch",
        "horizon": 64,
        "soft_horizon": True,
        "lr": 1e-4,
        "lr_schedule": [(0, 1e-4), (200000, 1e-8)],
        "train_batch_size": TRAIN_BATCH_SIZE,
        "lambda": 0.95,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "num_workers": NUM_WORKERS,
        "sgd_minibatch_size": 64 * 16,
        "rollout_fragment_length": ROLLOUT,  # 4C
        "num_gpus": 0.75,
        "num_sgd_iter": 3,
        "model": {
            "custom_model": "grid_actor_critic",
            "custom_model_config": {
                "num_tls": 3 * 3,
            },
        },
    }
)
save_root = "outputs/reproduce/rllib"
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
for n in range(NUM_EPISODES):
    result = trainer.train()
    save_file = trainer.save(save_root)
    print(
        status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            save_file
        )
    )
