import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ponder"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Taxi-v3"
    """the id of the environment"""
    total_timesteps: int = 5_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.95
    """the discount factor gamma"""
    alpha: float = 0.1
    """the learning rate of the q optimizer"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    num_planning_steps: int = 50
    """the number of planning steps"""


def make_env(env_id, seed, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=f"dyna-q_{args.env_id}_{args.num_planning_steps}",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    env = make_env(args.env_id, args.seed, args.capture_video, run_name)()

    assert isinstance(
        env.observation_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    model = {}
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_table[obs]
            action = np.argmax(q_values, axis=-1)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, truncated, info = env.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if terminated or truncated:
            if "episode" in info:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )

        q_table[obs, action] += args.alpha * (
            reward
            + args.gamma * np.max(q_table[next_obs]) * (1 - terminated)
            - q_table[obs, action]
        )
        model[(obs, action)] = (next_obs, reward, terminated)

        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)
        else:
            obs = next_obs

        for _ in range(args.num_planning_steps):
            random_obs, random_action = random.choice(list(model.keys()))

            next_obs, reward, terminated = model[(random_obs, random_action)]
            q_table[random_obs, random_action] += args.alpha * (
                reward
                + args.gamma * np.max(q_table[next_obs] * (1 - terminated))
                - q_table[random_obs, random_action]
            )

    env.close()
    writer.close()
