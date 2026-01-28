import argparse
import os
import sys
import numpy as np
import torch
import hydra
from src.utils.hydra import init_config
from src.data import Data, InstanceData
from src.transforms import instantiate_transforms

# Add project root to path
sys.path.append(os.getcwd())

def read_cloud(path):
    """Read point cloud from file."""
    print(f"Reading file: {path}")
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.txt':
        # Assume S3DIS format: x y z r g b (and maybe label, but we ignore it)
        try:
            data = np.loadtxt(path)
        except Exception as e:
            print(f"Error reading txt file: {e}")
            sys.exit(1)
            
        if data.shape[1] < 3:
            raise ValueError("File must have at least 3 columns (x, y, z)")
            
        pos = data[:, :3]
        if data.shape[1] >= 6:
            rgb = data[:, 3:6]
        else:
            rgb = np.zeros_like(pos) # Default black if no color
            
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    elif ext == '.npy':
        data = np.load(path)
        pos = data[:, :3]
        if data.shape[1] >= 6:
            rgb = data[:, 3:6]
        else:
            rgb = np.zeros_like(pos)
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    elif ext == '.ply':
        try:
            from plyfile import PlyData
        except ImportError:
            print("Please install plyfile to read .ply files: pip install plyfile")
            sys.exit(1)
            
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        pos = np.stack([x, y, z], axis=1)
        
        if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
            r = vertex['red']
            g = vertex['green']
            b = vertex['blue']
            rgb = np.stack([r, g, b], axis=1)
        else:
            rgb = np.zeros_like(pos)
            
        return Data(pos=torch.from_numpy(pos).float(), rgb=torch.from_numpy(rgb).float())
        
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def save_cloud(path, nag):
    """Save labeled point cloud to txt."""
    print(f"Saving results to: {path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # We save level-0 (points)
    # Note: These are subsampled points if GridSampling was used.
    data = nag[0]
    
    pos = data.pos.cpu().numpy()
    rgb = data.rgb.cpu().numpy() if data.rgb is not None else np.zeros_like(pos)
    
    # Check RGB range and scale if necessary
    if rgb.max() <= 1.0:
        print("Detected normalized RGB, scaling to 0-255")
        rgb = rgb * 255
    
    # Semantic prediction
    if data.semantic_pred is not None:
        pred = data.semantic_pred.cpu().numpy()
        if pred.ndim > 1:
            pred = np.argmax(pred, axis=1)
    else:
        pred = np.zeros(pos.shape[0])
        
    # Instance prediction (if available)
    obj_pred = np.zeros(pos.shape[0]) - 1 # -1 for no instance
    
    if hasattr(data, 'obj_pred') and data.obj_pred is not None:
        if isinstance(data.obj_pred, InstanceData):
            # InstanceData is a sparse structure. Convert to dense.
            # indices: point indices (group indices repeated by size)
            # values[0]: object IDs
            try:
                indices = data.obj_pred.indices.cpu().numpy()
                # Access obj via values list. InstanceData values order: obj, count, y
                # See src/data/instance.py __value_keys__
                # or values[0] if we assume standard init order
                obj_ids = data.obj_pred.values[0].cpu().numpy()
                
                # Reset to -1
                obj_pred[:] = -1
                obj_pred[indices] = obj_ids
            except Exception as e:
                print(f"Error processing InstanceData: {e}")
        elif isinstance(data.obj_pred, torch.Tensor):
            obj_pred = data.obj_pred.cpu().numpy()
        elif isinstance(data.obj_pred, np.ndarray):
            obj_pred = data.obj_pred
            
    # Combine: x y z r g b semantic_label instance_label
    # CloudCompare expects Scalar Field for label
    output_data = np.column_stack([pos, rgb, pred, obj_pred])
    
    # Format: xyz float, rgb int, labels int
    fmt = '%.6f %.6f %.6f %d %d %d %d %d'
    try:
        np.savetxt(path, output_data, fmt=fmt)
    except Exception as e:
        print(f"Error saving with format: {e}. Falling back to default.")
        np.savetxt(path, output_data)
        
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Inference on a single room file")
    parser.add_argument('--input', required=True, help='Input point cloud file (.txt, .ply, .npy)')
    parser.add_argument('--output', default='output/result_test.txt', help='Output txt file')
    parser.add_argument('--ckpt', default='/workspace/exos_8t_0/jts/OpenIns3D_new/superpoint_transformer/logs/train/runs/2025-12-03_12-51-22/checkpoints/epoch_759.ckpt', help='Checkpoint path')
    args = parser.parse_args()
    
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_pred.txt"
        
    # 1. Load Config
    # Use init_config from src.utils.hydra
    # Set PROJECT_ROOT environment variable for Hydra config resolution
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    
    cfg = init_config(config_name="train.yaml", overrides=[
        "experiment=panoptic/s3dis_with_stuff",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={args.ckpt}"
    ])
    
    # 2. Instantiate Datamodule to get transforms
    print("Instantiating datamodule (for transforms)...")
    # datamodule = hydra.utils.instantiate(cfg.datamodule)
    # S3DISDataModule uses pre_transform defined in config.
    # Since we are reading raw files, we need to apply pre_transform manually.
    # val_transform is usually None for S3DIS as it relies on pre-processed data.
    
    pre_transform_cfg = cfg.datamodule.get('pre_transform')
    if pre_transform_cfg is not None:
        print("Instantiating pre_transform...")
        pre_transform = instantiate_transforms(pre_transform_cfg)
    else:
        pre_transform = None
        
    # We also need on_device_transform. 
    # We can get it from instantiating datamodule or manually instantiating from config.
    # Instantiating datamodule is safer to get all properties.
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    on_device_transform = datamodule.on_device_val_transform
    
    # 3. Instantiate Model
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 4. Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    
    # 5. Read Input
    nag = read_cloud(args.input)
    
    # 6. Transform
    print("Applying transforms...")
    if pre_transform:
        print("Applying pre_transform...")
        nag = pre_transform(nag)
        
    # val_transform is usually None, but if it exists apply it
    if hasattr(datamodule, 'val_transform') and datamodule.val_transform:
        print("Applying val_transform...")
        nag = datamodule.val_transform(nag)
    
    # Move to device
    nag = nag.to(device)
    
    if on_device_transform:
        nag = on_device_transform(nag)
        
    # 7. Inference
    print("Running inference...")
    if hasattr(model, 'net'):
        model.net.store_features = True # Required for some outputs?
        
    with torch.no_grad():
        output = model(nag)
        
    # 8. Process Output
    nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
    
    # Panoptic/Instance prediction
    try:
        # For panoptic, we get semantic, instance_index, and instance_pred
        # voxel_panoptic_pred returns: (y_pred, index_pred, obj_pred)
        # Check source code of PanopticSegmentationOutput if possible
        res = output.voxel_panoptic_pred(super_index=nag[0].super_index)
        if isinstance(res, tuple) and len(res) == 3:
            _, _, vox_obj_pred = res
            nag[0].obj_pred = vox_obj_pred
            print("Panoptic predictions computed.")
        else:
            print("voxel_panoptic_pred returned unexpected format.")
    except Exception as e:
        print(f"Could not compute panoptic predictions: {e}")
        # If model is semantic only, this is expected.
    
    # 9. Save
    save_cloud(args.output, nag)

if __name__ == "__main__":
    main()
