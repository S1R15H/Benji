import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent.model import BenjiAgent

def main():
    parser = argparse.ArgumentParser(description="Train Benji Bananas RL Agent")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--save_freq", type=int, default=100000, help="Checkpoint frequency")
    parser.add_argument("--model", type=str, default=None, help="Path to existing model to load")
    parser.add_argument("--tensorboard", type=str, default="./logs/", help="Tensorboard log dir")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning Rate")
    
    args = parser.parse_args()
    
    print("Initializing Agent...")
    agent = BenjiAgent(
        model_path=args.model,
        tensorboard_log=args.tensorboard,
        learning_rate=args.lr
    )
    
    try:
        agent.train(
            total_timesteps=args.steps, 
            save_freq=args.save_freq
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "benji_ppo_interrupted")
        agent.model.save(path)
        
        # Save VecNormalize stats if present
        from stable_baselines3.common.vec_env import VecNormalize
        if hasattr(agent, "venv") and isinstance(agent.venv, VecNormalize):
            agent.venv.save(path + "_vecnormalize.pkl")
        elif hasattr(agent, "env") and isinstance(agent.env, VecNormalize):
             agent.env.save(path + "_vecnormalize.pkl")
            
        print(f"Saved to {path}.zip")
    finally:
        agent.close()

if __name__ == "__main__":
    main()
