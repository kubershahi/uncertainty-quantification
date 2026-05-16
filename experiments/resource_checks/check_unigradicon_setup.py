import torch
import torch.nn.functional as F
import numpy as np
from unigradicon import get_unigradicon

def preprocess(img, img_type="mri"):
    """Exact preprocessing from the official demo"""
    if img_type == "ct":
        clamp = [-1000, 1000]
        img = (torch.clamp(img, clamp[0], clamp[1]) - clamp[0]) / (clamp[1] - clamp[0])
    elif img_type == "mri":
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img - im_min) / (im_max - im_min)
    
    # UniGradICON MUST have [175, 175, 175] resolution
    return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)

def run_check():
    print("--- UniGradICON Official Demo Check ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print("Loading model weights...")
    net = get_unigradicon()
    net.to(device)
    net.eval()

    # Create Dummy Data (Simulating a raw medical volume)
    # Shape: (Batch=1, Channel=1, Depth, Height, Width)
    raw_data = torch.randn(1, 1, 100, 100, 100) 
    
    print("Preprocessing inputs to 175x175x175...")
    source = preprocess(raw_data.clone(), img_type="mri").to(device)
    target = preprocess(raw_data.clone(), img_type="mri").to(device)

    print("Running inference...")
    with torch.no_grad():
        # The model stores results internally; it does not return them here.
        net(source, target)

    # Access the results from the class attributes as shown in the demo
    warped = net.warped_image_A
    phi = net.phi_AB_vectorfield

    print("Success: Inference complete.")
    print(f"Warped image shape: {warped.shape}")
    print(f"Vector field shape: {phi.shape}")
    print("--- Setup is VERIFIED ---")

if __name__ == "__main__":
    run_check()