import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.model import BenjiAgent

class GradCAM:
    """
    Grad-CAM implementation for the BenjiAgent's CustomCNN.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = grad_output[0]

    def __call__(self, obs_tensor: torch.Tensor, action_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for the given observation.
        """
        # Zero grads
        self.model.policy.zero_grad()
        
        # Forward pass through the policy
        # interacting directly with the policy network to get logits/values
        # PPO Policy: features_extractor -> mlp_extractor -> action_net / value_net
        
        features = self.model.policy.features_extractor(obs_tensor)
        latent_pi, latent_vf = self.model.policy.mlp_extractor(features)
        distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
        
        # We want to visualize what contributes to the CHOSEN action (or max prob action)
        if action_idx is None:
            # If no action provided, use the one with highest probability (mode)
            action = distribution.mode()
            action_idx = action.item()
        
        # Get the log probability of the selected action as the target to maximize/explain
        # One option: distribution.log_prob(action)
        # Another option: raw logits (if available, mostly categorical) -- SB3 keeps distributions abstract
        
        # For simplicity in PPO (Categorical), log_prob is differentiable.
        log_prob = distribution.log_prob(torch.tensor([action_idx]).to(obs_tensor.device))
        
        # Backward pass from the log_prob
        log_prob.backward()
        
        # Generate Heatmap
        # GAP (Global Average Pooling) of gradients
        # gradients shape: [Batch, Channels, H, W]
        # We assume batch size 1 for visualization
        gradients = self.gradients
        activations = self.activations
        
        # Mean over H,W dims (2,3) -> [Batch, Channels]
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        
        # Weight the activations
        # activations: [Batch, Channels, H, W]
        # pooled_gradients: [Batch, Channels]
        # weighted: [Batch, Channels, H, W]
        # We use broadcasting: [Batch, Channels, 1, 1]
        weighted_activations = activations * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
        
        # Average the channels of the weighted activations (Linear combination)
        # Result: [Batch, H, W]
        heatmap = torch.mean(weighted_activations, dim=1)
        
        # ReLU to keep positive contributions
        heatmap = F.relu(heatmap)
        
        # Convert to numpy and take first in batch
        heatmap = heatmap[0].cpu().detach().numpy()
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, action_idx

def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlays the heatmap on the image.
    """
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB using colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def main():
    parser = argparse.ArgumentParser(description="Benji Bananas Saliency Map Viewer")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
    parser.add_argument("--save_dir", type=str, default="saliency_logs", help="Directory to save visualized frames")
    parser.add_argument("--image", type=str, help="Path to a single image file to visualize. If set, ignores episodes/env.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        print(f"Error: Model not found at {args.model}")
        return

    # Load Agent
    print(f"Loading Agent from {args.model}...")
    # Use offline mode if just visualizing an image to avoid device connection
    is_offline = args.image is not None
    agent = BenjiAgent(model_path=args.model, offline=is_offline)
    
    # Setup Grad-CAM
    # Target Layer: index 6 of CNN (last conv layer before flatten)
    target_layer = agent.model.policy.features_extractor.cnn[6]
    grad_cam = GradCAM(agent.model, target_layer)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Starting Saliency Visualization...")

    if args.image:
        # Visualize single image
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
            
        print(f"Processing single image: {args.image}")
        # Load and preprocess
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print("Failed to load image.")
            return
            
        # Use BenjiPreprocessor manually
        from env.preprocessing import BenjiPreprocessor
        preprocessor = BenjiPreprocessor()
        processed = preprocessor.process_frame(img_bgr) # (1, 128, 128)
        
        # Stack 4 times to match env observation space
        # (1, 4, 128, 128)
        obs = np.repeat(processed[np.newaxis, ...], 4, axis=1)
        
        # Handle VecNormalize
        from stable_baselines3.common.vec_env import VecNormalize
        if isinstance(agent.venv, VecNormalize):
             print("Normalizing observation using loaded VecNormalize stats...")
             # normalize_obs ensures the input matches what the model expects
             obs = agent.venv.normalize_obs(obs)
        
        # Run Grad-CAM
        obs_tensor = torch.as_tensor(obs).to(agent.model.device).float()

        
        # We need to ensure normalisation matches training.
        # If model used VecNormalize, we should ideally normalize this obs.
        # But we don't have easy access to the exact running mean/std here unless we use agent.venv.
        # agent.venv is a DummyVecEnv wrapping BenjiBananasEnv.
        # If loaded from file, it might have VecNormalize.
        # Let's try to pass it through agent.venv if possible?
        # A bit tricky manually.
        # For now, assume raw observation or simple scaling.
        
        heatmap, action = grad_cam(obs_tensor)
        
        # Visualize
        # Use simple resizing of original image for background?
        # Or use the processed frame?
        # Let's use the original image resized to 128x128 or keep original size?
        # Overlay heatmap uses the img passed to it.
        # Let's resize original to reasonable size (e.g. keep aspect if possible or just use what we have)
        
        # We'll use the preprocessed frame for the "Agent's View" visualization
        agent_view = processed[0] # (128, 128)
        agent_view_bgr = cv2.cvtColor(agent_view, cv2.COLOR_GRAY2BGR)
        
        viz = overlay_heatmap(agent_view_bgr, heatmap)
        
        filename = os.path.join(args.save_dir, f"saliency_{os.path.basename(args.image)}_act{action}.png")
        cv2.imwrite(filename, viz)
        print(f"Saved {filename}")
        return

    # Online / Episode Loop
    env = agent.venv
    try:
        for ep in range(args.episodes):
            # Retry loop to ensure we get a meaningful episode
            max_retries = 5
            for attempt in range(max_retries):
                obs = env.reset()
                done = False
                step = 0
                
                # Setup Video Writer
                video_path = os.path.join(args.save_dir, f"saliency_ep{ep}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_size = (512, 512) # Upscaled
                out = cv2.VideoWriter(video_path, fourcc, 30.0, out_size)
                print(f"Recording video to {video_path} (Attempt {attempt+1})...")
                
                while not done:
                    # Prepare tensor for Grad-CAM
                    obs_tensor = torch.as_tensor(obs).to(agent.model.device).float()
                    
                    # Get the heatmap
                    heatmap, action = grad_cam(obs_tensor)
                    
                    # Step the environment
                    obs, rewards, dones, infos = env.step(np.array([action]))
                    
                    done = dones[0]
                    
                    # Visualization
                    frame = obs[0, -1, :, :] # Last frame
                    
                    # Normalize for display
                    if frame.min() < 0 or frame.max() > 255:
                         frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
                    
                    frame = frame.astype(np.uint8)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    viz = overlay_heatmap(frame_rgb, heatmap)
                    
                    # Add Action Text
                    action_text = "HOLD" if action == 1 else "RELEASE"
                    color = (0, 255, 0) if action == 1 else (0, 0, 255)
                    cv2.putText(viz, action_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Upscale for video
                    viz_large = cv2.resize(viz, out_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Write to video
                    out.write(viz_large)
                    
                    if step % 50 == 0:
                        print(f"Ep {ep} | Step {step}", end='\r')
                    
                    step += 1
                    
                    # Early stop for debugging/safety
                    if step > 2000: 
                        break
                
                out.release()
                
                if step > 15:
                    print(f"\nSaved {video_path} ({step} steps)")
                    break # Success, move to next episode index
                else:
                     print(f"\nEpisode too short ({step} steps). Discarding and retrying...")

    except KeyboardInterrupt:
        print("Stopping...")
        if 'out' in locals():
            out.release()

if __name__ == "__main__":
    main()
