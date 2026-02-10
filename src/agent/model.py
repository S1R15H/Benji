from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Optional
import os

from env.benji_env import BenjiBananasEnv
from agent.callbacks import TensorboardCallback, PauseCallback

from stable_baselines3.common.monitor import Monitor

class CustomCNN(BaseFeaturesExtractor):
    """
    Deeper CNN for 128x128 resolution.
    Structure:
    - Conv1: 32, 8x8, 4
    - Conv2: 64, 4x4, 2
    - Conv3: 64, 3x3, 1
    - Conv4: 128, 3x3, 1 (New)
    - Flatten -> Linear(512)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by passing dummy
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class BenjiAgent:
    """
    Wrapper for the PPO agent trained on Benji Bananas.
    Handles environment wrapping (Stacking) and Model initialization.
    """
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 tensorboard_log: str = "./logs/",
                 offline: bool = False,
                 learning_rate: float = 2.5e-4):

        
        # 1. Setup Environment
        # We need to wrap the raw Env to handle Frame Stacking (4 frames)
        # We also need Monitor to track Episode Stats for Tensorboard.
        self.env = BenjiBananasEnv(offline=offline)
        self.env = Monitor(self.env) # Add Monitor Wrapper
        
        if offline:
             # In offline mode, maybe we strictly need DummyVecEnv
             pass
        
        self.venv = DummyVecEnv([lambda: self.env])
        # self.venv = VecTransposeImage(self.venv) # Removed: Env now outputs (C, H, W) directly
        self.venv = VecFrameStack(self.venv, n_stack=4)
        

        # Handle VecNormalize Loading/Creation
        stats_path = model_path.replace(".zip", "_vecnormalize.pkl") if model_path else None
        
        if stats_path and os.path.exists(stats_path):
            print(f"Loading VecNormalize stats from {stats_path}")
            # Load wraps the *source* env (self.venv so far)
            self.venv = VecNormalize.load(stats_path, self.venv)
            # Important: Set training mode implies updating stats.
            # If we are fine-tuning, yes. If evaluating, maybe no. 
            # Assuming fine-tuning/training.
            self.venv.training = True 
            self.venv.norm_obs = True
            self.venv.norm_reward = True
        else:
            # Create fresh
            self.venv = VecNormalize(self.venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

        # 2. Initialize Model
        self.continue_training = False
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = PPO.load(model_path, env=self.venv, tensorboard_log=tensorboard_log)
            self.continue_training = True
            
            # Override Learning Rate for Fine-tuning
            print(f"Setting Learning Rate to {learning_rate}")
            from stable_baselines3.common.utils import get_schedule_fn
            self.model.learning_rate = learning_rate
            self.model.lr_schedule = get_schedule_fn(learning_rate)
        else:
            print("Initializing new PPO Agent (CustomCNN + VecNormalize)...")
            # Hyperparameters from Phase 3 SOTA Plan (Tuned for Speed)
            self.model = PPO(
                "CnnPolicy",
                self.venv,
                verbose=1,
                learning_rate=learning_rate,
                n_steps=2048,   # Rollout Buffer Size (Collect-Then-Train)
                batch_size=128, # Mini-batch size
                n_epochs=10,    # Review buffer 10 times
                gamma=0.99, 
                clip_range=0.2,
                ent_coef=0.05, # Increased to 0.05
                tensorboard_log=tensorboard_log,
                policy_kwargs={
                    "features_extractor_class": CustomCNN,
                    "features_extractor_kwargs": {"features_dim": 512},
                    "normalize_images": True
                }
            )


    def train(self, total_timesteps: int = 100000, save_freq: int = 10000, save_path: str = "./models/"):
        """
        Executes the training loop.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Checkpoint Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="benji_ppo",
            save_vecnormalize=True 
        )
        
        # 2. Tensorboard Logging Callback
        tb_callback = TensorboardCallback()
        
        # 3. Pause Callback (For Collect-Then-Train)
        pause_callback = PauseCallback()
        
        # Combine callbacks
        callbacks = [checkpoint_callback, tb_callback, pause_callback]
        
        print(f"Starting training for {total_timesteps} steps...")
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=callbacks,
            reset_num_timesteps=not self.continue_training
        )
        print("Training complete.")
        
        final_path = os.path.join(save_path, "benji_ppo_final")
        self.model.save(final_path)
        # Also save normalization stats
        if isinstance(self.venv, VecNormalize):
             self.venv.save(os.path.join(save_path, "benji_ppo_final_vecnormalize.pkl"))
             
        print(f"Model saved to {final_path}")

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
        
    def close(self):
        self.venv.close()
