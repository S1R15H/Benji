import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agent.model import BenjiAgent
from agent.dataset import BenjiBCDataset

def train_bc(epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = BenjiBCDataset()
    if len(dataset) == 0:
        print("No data found! Run collector.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Init Agent (Offline)
    print("Initializing Agent...")
    agent = BenjiAgent(offline=True)
    policy = agent.model.policy.to(device)
    
    # 3. Setup Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # 4. Training Loop
    print(f"Starting BC Training for {epochs} epochs...")
    policy.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (obs, actions) in enumerate(dataloader):
            obs = obs.float().to(device) # (B, 4, 84, 84)
            actions = actions.to(device) # (B)
            
            # Forward Pass
            # SB3 Policy returns distribution
            # We assume NatureCNN normalizes if configured (it is)
            dist = policy.get_distribution(obs)
            
            # Loss: Negative Log Likelihood
            loss = -dist.log_prob(actions).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred_actions = dist.mode() # discrete: argmax
            correct += (pred_actions == actions).sum().item()
            total += actions.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {correct/total:.2%}", end='\r')
                
        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"\nEpoch {epoch+1} Done. Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2%}")
        
    # 5. Save
    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_bc_pretrained"
    agent.model.save(save_path)
    print(f"Pre-trained model saved to {save_path}.zip")

if __name__ == "__main__":
    train_bc()
