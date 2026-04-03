"""
检查 batch_inference.py 中错误预测点显示问题的根本原因

问题分析：
1. visualize_3d 函数需要 data_0.y 和 data_0.semantic_pred 都存在才能显示错误
2. batch_inference.py 在推理时会设置 semantic_pred
3. 关键问题：数据加载时是否保留了 ground truth 标签 'y'

关键代码位置：
- src/datasets/base.py 第1098-1104行：NAG.load() 使用 point_load_keys 参数
- src/data/nag.py 第522-524行：如果 keys_low=None，会加载所有键（包括 'y'）
- configs/datamodule/semantic/default.yaml：没有设置 point_load_keys

结论：
如果 point_load_keys=None（默认），数据加载时会包含 'y' 标签。
但是，可能在 on_device_transform 中丢失了 'y' 标签！

让我检查 on_device_transform 中的 transforms...
"""

import os
import sys

# 添加项目路径
sys.path.append(os.getcwd())
os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

def check_on_device_transforms():
    """检查 on_device_transform 是否会移除 'y' 标签"""
    
    print("=" * 80)
    print("检查 on_device_transform 中的 transforms")
    print("=" * 80)
    
    # 读取配置文件
    import yaml
    
    config_path = "configs/datamodule/semantic/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n[检查 on_device_val_transform]")
    print("\n包含的 transforms:")
    
    if 'on_device_val_transform' in config:
        transforms = config['on_device_val_transform']
        if transforms:
            for i, t in enumerate(transforms):
                transform_name = t.get('transform', 'Unknown')
                params = t.get('params', {})
                
                print(f"\n  [{i}] {transform_name}")
                if params:
                    print(f"      参数: {params}")
                
                # 检查是否会移除 'y' 标签
                if 'Remove' in transform_name or 'Select' in transform_name:
                    if 'keys' in params:
                        keys = params['keys']
                        if isinstance(keys, str):
                            keys = [keys]
                        if 'y' in keys:
                            print(f"      ⚠️  警告: 此 transform 会移除 'y' 标签！")
                        else:
                            print(f"      ✅ 此 transform 不会影响 'y' 标签")
                elif 'Cast' in transform_name or 'NAGCast' in transform_name:
                    print(f"      ✅ 此 transform 只是类型转换，不会移除 'y' 标签")
                else:
                    print(f"      ℹ️  此 transform 可能不影响 'y' 标签")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)
    
    print("""
根据代码分析，on_device_val_transform 中的 transforms 不会主动移除 'y' 标签。

但是，可能存在以下问题：

1. **NAGCast transform**:
   - 这个 transform 会将所有属性转换为 float 或 long
   - 如果 'y' 标签的 dtype 不正确，可能会导致问题

2. **数据预处理问题**:
   - 在 pre_transform 中，GridSampling3D 使用 hist_key='y' 参数
   - 这会将 'y' 转换为直方图格式
   - 可能在某个环节丢失了 'y' 属性

3. **可能的解决方案**:
   - 检查数据加载后是否真的有 'y' 属性
   - 在可视化前添加调试代码打印 data_0.keys
   - 确保 'y' 属性在所有 transforms 后仍然存在

建议的调试方法：
在 batch_inference.py 的 generate_visualization 函数中添加：
    print(f"Level-0 keys: {nag[0].keys}")
    print(f"Has 'y': {'y' in nag[0].keys}")
    print(f"Has 'semantic_pred': {'semantic_pred' in nag[0].keys}")
    """)

if __name__ == "__main__":
    check_on_device_transforms()
