from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the 'info' dict of the first environment
        # The 'infos' are accessible via self.locals['infos']
        infos = self.locals.get("infos", [])
        if infos:
             # Aggregate components from all envs (if parallel) or just take first
             # For now, taking first is fine for DummyVecEnv(1)
            info = infos[0]
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    self.logger.record(f"custom/{key}", value)
        return True

class PauseCallback(BaseCallback):
    """
    A custom callback that pauses the environment during training (updating gradients)
    and unpauses it during rollouts (data collection).
    
    This is critical for Real-Time environments where the game keeps running
    even if the agent is 'thinking' or 'updating'.
    """
    def __init__(self, verbose=0):
        super(PauseCallback, self).__init__(verbose)

    def _on_rollout_start(self) -> None:
        """
        Triggered before collecting new samples. 
        Game should be RUNNING.
        """
        # Access the underlying environment
        # self.training_env is a VecEnv.
        # We need to call unpause() on the actual BenjiBananasEnv instance.
        # Since it's wrapped in Monitor -> VecFrameStack -> dummyVecEnv...
        # We try to access methods via 'env_method'.
        
        logger.info("[Callback] Rollout Started. Unpausing Game...")
        try:
            self.training_env.env_method("unpause")
        except Exception as e:
            logger.warning(f"Failed to unpause environment: {e}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy.
        Game should be PAUSED.
        """
        logger.info("[Callback] Rollout Ended. Pausing Game for Training...")
        try:
            self.training_env.env_method("pause")
        except Exception as e:
            logger.warning(f"Failed to pause environment: {e}")
