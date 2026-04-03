"""
诊断脚本：检查 batch_inference.py 中错误预测点显示问题
"""
import os
import sys
import torch
import hydra

sys.path.append(os.getcwd())
os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

from src.utils.hydra import init_config

def diagnose_scannet_inference():
    """诊断 ScanNet 推理中的标签加载问题"""
    
    print("=" * 80)
    print("诊断 ScanNet 推理中的错误预测点显示问题")
    print("=" * 80)
    
    # 1. 加载配置
    print("\n[步骤1] 加载配置...")
    cfg = init_config(config_name="train.yaml", overrides=[
        "experiment=panoptic/scannet",
        "datamodule.dataloader.batch_size=1",
        "datamodule.dataloader.num_workers=0",
    ])
    
    # 2. 实例化 datamodule
    print("\n[步骤2] 实例化 datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage='val')
    dataset = datamodule.val_dataset
    
    # 3. 检查数据集配置
    print("\n[步骤3] 检查数据集配置...")
    print(f"  - 数据集类型: {type(dataset).__name__}")
    print(f"  - 类别数量: {dataset.num_classes}")
    print(f"  - point_load_keys: {dataset.point_load_keys}")
    print(f"  - segment_load_keys: {dataset.segment_load_keys}")
    
    # 4. 加载一个样本并检查属性
    print("\n[步骤4] 加载样本并检查属性...")
    sample = dataset[0]
    
    print(f"  - 样本类型: {type(sample).__name__}")
    print(f"  - 样本级别数: {sample.num_levels if hasattr(sample, 'num_levels') else 'N/A'}")
    
    # 检查 level-0 (点级别) 的属性
    if hasattr(sample, 'num_levels'):
        data_0 = sample[0]
        print(f"\n  Level-0 (点级别) 属性:")
        print(f"    - 所有键: {list(data_0.keys)}")
        print(f"    - 是否有 'y': {'y' in data_0.keys}")
        print(f"    - 是否有 'pos': {'pos' in data_0.keys}")
        print(f"    - 是否有 'rgb': {'rgb' in data_0.keys}")
        
        if 'y' in data_0.keys:
            y = data_0.y
            print(f"    - 'y' 形状: {y.shape}")
            print(f"    - 'y' 类型: {type(y)}")
            print(f"    - 'y' dtype: {y.dtype if hasattr(y, 'dtype') else 'N/A'}")
            print(f"    - 'y' 取值范围: [{y.min() if hasattr(y, 'min') else 'N/A'}, {y.max() if hasattr(y, 'max') else 'N/A'}]")
            print(f"    - 'y' 唯一值数量: {len(y.unique()) if hasattr(y, 'unique') else 'N/A'}")
        else:
            print(f"    ❌ 关键问题: Level-0 没有 'y' 属性！")
    
    # 5. 检查 on_device_transform
    print("\n[步骤5] 检查 on_device_transform...")
    on_device_transform = getattr(datamodule, 'on_device_val_transform', None)
    if on_device_transform is not None:
        print(f"  - on_device_transform 类型: {type(on_device_transform).__name__}")
        print(f"  - 包含的 transforms:")
        for i, t in enumerate(on_device_transform.transforms):
            print(f"    [{i}] {type(t).__name__}")
            # 检查是否有移除键的操作
            if hasattr(t, 'keys'):
                print(f"        - keys: {t.keys}")
            if 'Remove' in type(t).__name__ or 'Select' in type(t).__name__:
                print(f"        ⚠️  可能会影响 'y' 属性")
    else:
        print("  - on_device_transform: None")
    
    # 6. 模拟推理流程
    print("\n[步骤6] 模拟推理流程...")
    print("  加载模型...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 模拟 checkpoint 加载（不实际加载权重）
    device = torch.device("cpu")
    model = model.eval().to(device)
    
    if hasattr(model, 'net'):
        model.net.store_features = True
    
    # 应用 on_device_transform
    print("  应用 on_device_transform...")
    if on_device_transform is not None:
        sample_transformed = on_device_transform(sample.to(device))
    else:
        sample_transformed = sample.to(device)
    
    # 检查变换后的属性
    if hasattr(sample_transformed, 'num_levels'):
        data_0_transformed = sample_transformed[0]
        print(f"\n  变换后 Level-0 属性:")
        print(f"    - 所有键: {list(data_0_transformed.keys)}")
        print(f"    - 是否有 'y': {'y' in data_0_transformed.keys}")
        
        if 'y' in data_0_transformed.keys:
            y_transformed = data_0_transformed.y
            print(f"    - 'y' 形状: {y_transformed.shape}")
            print(f"    ✅ 'y' 属性在变换后仍然存在")
        else:
            print(f"    ❌ 关键问题: 'y' 属性在 on_device_transform 后丢失！")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_scannet_inference()
