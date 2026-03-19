import argparse
import os
import sys
import numpy as np
import torch
import hydra
from src.utils.hydra import init_config
from src.data import Data, InstanceData
from src.transforms import instantiate_transforms
from src.visualization.visualization import visualize_3d

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
            
    # Extract multi-level superpoint indices
    sp_indices = []
    if isinstance(nag, list) or hasattr(nag, 'num_levels'):
        num_levels = getattr(nag, 'num_levels') if hasattr(nag, 'num_levels') else len(nag)
        current_sp = None
        for i in range(num_levels - 1):
            if hasattr(nag[i], 'super_index') and nag[i].super_index is not None:
                if i == 0:
                    current_sp = nag[i].super_index.cpu().numpy()
                else:
                    current_sp = nag[i].super_index.cpu().numpy()[current_sp]
                sp_indices.append(current_sp)
    
    # Combine: x y z r g b semantic_label instance_label [sp_level_1 ... sp_level_n]
    sp_columns = [sp.reshape(-1, 1) for sp in sp_indices]
    if sp_columns:
        output_data = np.column_stack([pos, rgb, pred, obj_pred] + sp_columns)
    else:
        output_data = np.column_stack([pos, rgb, pred, obj_pred])
    
    # Dynamic format string
    fmt_list = ['%.6f', '%.6f', '%.6f', '%d', '%d', '%d', '%d', '%d']
    fmt_list.extend(['%d'] * len(sp_indices))
    fmt = ' '.join(fmt_list)
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.ply':
        # Add PLY magic numbers and header for CloudCompare
        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {output_data.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int semantic_pred",
            "property int obj_pred"
        ]
        for i in range(len(sp_indices)):
            header_lines.append(f"property int sp_level_{i+1}")
        header_lines.append("end_header")
        header = "\n".join(header_lines)
        
        np.savetxt(path, output_data, fmt=fmt, header=header, comments='')
    else:
        try:
            np.savetxt(path, output_data, fmt=fmt)
        except Exception as e:
            print(f"Error saving with format: {e}. Falling back to default.")
            np.savetxt(path, output_data)
        
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Inference on a single room file")
    parser.add_argument('--input', required=True, help='Input point cloud file (.txt, .ply, .npy)')
    parser.add_argument('--output', default=None, help='Output txt file. By default, it creates <input_base>_pred.txt')
    parser.add_argument('--ckpt', default='/workspace/exos_8t_0/jts/OpenIns3D_new/superpoint_transformer/logs/train/runs/2026-01-29_13-54-08/checkpoints/epoch_199.ckpt', help='Checkpoint path')
    parser.add_argument('--experiment', default='panoptic/kitti360', help='Experiment name used flexibly for config. E.g., panoptic/s3dis, panoptic/kitti360, panoptic/scannet')
    parser.add_argument('--visualize', action='store_true', help='Visualize results with superpoints using Plotly and save as HTML')
    args = parser.parse_args()
    
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_pred.txt"
        
    # 1. Load Config
    # Use init_config from src.utils.hydra
    # Set PROJECT_ROOT environment variable for Hydra config resolution
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    
    cfg = init_config(config_name="train.yaml", overrides=[
        f"experiment={args.experiment}",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
        f"ckpt_path={args.ckpt}"
    ])
    
    # 动态解析数据集类别配置
    dataset_name = args.experiment.split('/')[-1].lower()
    if 's3dis' in dataset_name:
        from src.datasets.s3dis_config import CLASS_NAMES, CLASS_COLORS, S3DIS_NUM_CLASSES, STUFF_CLASSES, STUFF_CLASSES_MODIFIED
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = S3DIS_NUM_CLASSES
        stuff_classes = STUFF_CLASSES_MODIFIED if 'stuff' in dataset_name else STUFF_CLASSES
    elif 'kitti' in dataset_name:
        from src.datasets.kitti360_config import CLASS_NAMES, CLASS_COLORS, KITTI360_NUM_CLASSES, STUFF_CLASSES
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = KITTI360_NUM_CLASSES
        stuff_classes = STUFF_CLASSES
    elif 'scannet' in dataset_name:
        from src.datasets.scannet_config import CLASS_NAMES, CLASS_COLORS, SCANNET_NUM_CLASSES, STUFF_CLASSES
        class_names = CLASS_NAMES
        class_colors = CLASS_COLORS
        num_classes = SCANNET_NUM_CLASSES
        stuff_classes = STUFF_CLASSES
    else:
        print(f"Warning: Unknown dataset in experiment '{args.experiment}'. Visualization colors might be incomplete.")
        class_names, class_colors, num_classes, stuff_classes = None, None, None, None

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
    if getattr(nag, 'pos', None) is not None and nag.pos.shape[0] == 0:
        print(f"Error: The input point cloud '{args.input}' contains 0 points. Exiting.")
        sys.exit(1)
    
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
        model.net.store_features = True # Required for some outputs
        
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

    # 10. Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        # Move nag back to CPU for visualization
        nag = nag.to('cpu')
        
        vis_output = visualize_3d(
            nag,
            class_names=class_names,
            class_colors=class_colors,
            stuff_classes=stuff_classes,
            num_classes=num_classes,
            max_points=100000,
            centroids=True,     # 生成聚类超点中心可视化
            h_edge=True,        # 可视化超点划分边缘连接
            h_edge_width=2
        )
        
        output_html = os.path.splitext(args.output)[0] + "_vis.html"
        fig = vis_output['figure']

        # Update layout to be responsive and full screen
        fig.update_layout(
            autosize=True,
            width=None,
            height=None,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        fig.write_html(output_html, config={'responsive': True})
        print(f"Visualization HTML saved to {output_html}")


if __name__ == "__main__":
    main()
