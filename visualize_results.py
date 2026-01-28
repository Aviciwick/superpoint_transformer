import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
import os
import sys

import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.getcwd())

from src.visualization.visualization import visualize_3d
from src.utils.hydra import init_config

def main():
    # 1. Initialize Hydra and load config
    # We assume we are running from project root
    cfg = init_config(config_name="train.yaml", overrides=[
        "experiment=panoptic/s3dis_with_stuff",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        "ckpt_path=/workspace/exos_8t_0/jts/superpoint_transformer/logs/train/runs/2025-12-03_12-51-22/checkpoints/epoch_759.ckpt" 
    ])

    # 2. Instantiate Datamodule
    print("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.val_dataset # Use validation dataset for visualization

    # 3. Instantiate Model
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 4. Load Checkpoint
    print(f"Loading checkpoint from {cfg.ckpt_path}...")
    # We load state dict directly since model is already instantiated with complex args
    checkpoint = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # 5. Load a sample
    print("Loading sample...")
    nag = dataset[0]
    
    # 6. Transform and Inference
    print("Running inference...")
    # For visualization, we want to store features
    if hasattr(model, 'net'):
        model.net.store_features = True
        
    nag = dataset.on_device_transform(nag.to(device))
    
    with torch.no_grad():
        output = model(nag)
        
    # 7. Process output
    print("Processing output...")
    # Compute voxel predictions
    nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
    
    # Panoptic predictions
    # Check if panoptic task
    # cfg.experiment string contains 'panoptic'
    # But we can just try/except or check model type
    try:
        vox_y, vox_index, vox_obj_pred = output.voxel_panoptic_pred(super_index=nag[0].super_index)
        nag[0].obj_pred = vox_obj_pred
        print("Panoptic predictions computed.")
    except AttributeError:
        print("Model does not support panoptic prediction or method missing.")

    # 8. Visualize
    print("Generating visualization...")
    # Move nag back to CPU for visualization
    nag = nag.to('cpu')
    
    output = visualize_3d(
        nag,
        class_names=dataset.class_names,
        class_colors=dataset.class_colors,
        stuff_classes=dataset.stuff_classes,
        num_classes=dataset.num_classes,
        max_points=100000,
        centroids=True,
        h_edge=True,
        h_edge_width=2
    )
    
    # 9. Save HTML
    output_html = "visualization.html"
    fig = output['figure']
    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")

if __name__ == "__main__":
    main()
