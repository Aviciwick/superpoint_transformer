"""
快速测试脚本：验证错误预测点显示功能

使用方法：
    python test_error_visualization.py

预期结果：
    生成 test_error_visualization.html，其中包含红色错误预测点
"""
import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

from src.data import Data, NAG
from src.visualization.visualization import visualize_3d

def test_error_visualization():
    """测试错误预测点可视化功能"""
    
    print("=" * 80)
    print("测试错误预测点可视化功能")
    print("=" * 80)
    
    # 创建测试数据
    num_points = 5000
    num_classes = 20
    
    print(f"\n创建测试数据:")
    print(f"  - 点数: {num_points}")
    print(f"  - 类别数: {num_classes}")
    
    # 创建随机点云
    pos = torch.randn(num_points, 3)
    
    # 创建 ground truth 标签
    y = torch.randint(0, num_classes, (num_points,))
    
    # 创建预测标签（故意引入错误）
    semantic_pred = y.clone()
    
    # 随机选择 20% 的点设置错误预测
    num_errors = int(num_points * 0.2)
    error_indices = np.random.choice(num_points, num_errors, replace=False)
    
    for idx in error_indices:
        # 设置一个不同的预测标签
        wrong_label = (y[idx] + torch.randint(1, num_classes, (1,))) % num_classes
        semantic_pred[idx] = wrong_label
    
    print(f"  - 错误预测点数: {num_errors} ({num_errors/num_points*100:.1f}%)")
    
    # 创建 Data 对象
    data = Data(
        pos=pos,
        y=y,
        semantic_pred=semantic_pred
    )
    
    # 创建 NAG 对象
    nag = NAG([data])
    
    print(f"\n检查数据属性:")
    print(f"  - Level-0 键: {list(nag[0].keys)}")
    print(f"  - 是否有 'y': {'y' in nag[0].keys}")
    print(f"  - 是否有 'semantic_pred': {'semantic_pred' in nag[0].keys}")
    
    if 'y' in nag[0].keys:
        print(f"  - 'y' 形状: {nag[0].y.shape}, dtype: {nag[0].y.dtype}")
    
    if 'semantic_pred' in nag[0].keys:
        print(f"  - 'semantic_pred' 形状: {nag[0].semantic_pred.shape}, dtype: {nag[0].semantic_pred.dtype}")
    
    # 可视化
    print(f"\n生成可视化...")
    output = visualize_3d(
        nag,
        num_classes=num_classes,
        max_points=num_points,
        point_size=3
    )
    
    # 保存 HTML
    output_file = "test_error_visualization.html"
    output['figure'].write_html(output_file, config={'responsive': True})
    
    print(f"\n✅ 测试完成!")
    print(f"📄 请打开 '{output_file}' 查看结果")
    print(f"\n预期结果:")
    print(f"  1. HTML 中应该有一个 'Semantic Errors' 按钮")
    print(f"  2. 点击按钮后，错误预测的点应该以红色显示")
    print(f"  3. 应该有约 {num_errors} 个红色错误点")
    
    # 验证错误检测逻辑
    print(f"\n验证错误检测逻辑:")
    y_np = y.numpy()
    pred_np = semantic_pred.numpy()
    detected_errors = np.where(pred_np != y_np)[0]
    print(f"  - 检测到的错误点数: {len(detected_errors)}")
    print(f"  - 预期的错误点数: {num_errors}")
    print(f"  - 匹配: {'✅' if len(detected_errors) == num_errors else '❌'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_error_visualization()
