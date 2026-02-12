这是 **H-SPT** 系统的最后一公里：**模块三：残差细化模块 (Residual Refinement Module, RRM)**。

如果说模块一是“安检员”（筛选难例），模块二是“充电站”（增强特征），那么模块三就是**“外科医生”**。它负责在微观层面进行手术，修复由于超点划分导致的锯齿状边缘，实现**“亚超点级 (Sub-superpoint)”** 的分割精度。

以下是专为 **AI Code Agent** 准备的详细设计方案。

---

# 模块三需求规格说明书：残差细化模块 (RRM)

**项目代号**：H-SPT (Hybrid-Superpoint Transformer)
**模块名称**：Module 3 - ResidualRefinementModule
**设计范式**：Point-Level Residual Learning (Shared MLP)

## 1. 模块功能定义 (Functional Specification)

本模块的核心任务是**“打破超点边界 (Superpoint Breaking)”**。
它接收模块二增强后的超点特征和原始点特征，通过逐点（Point-wise）分类，预测每个原始点的语义标签偏移量，从而修正超点内部的分类错误。

**核心逻辑链**：
1.  **特征广播 (Broadcasting)**：将宏观的“超点特征”复制 $N$ 份，对齐到微观的“原始点”。
2.  **特征拼接 (Concatenation)**：将“超点上下文”与“原始点细节”结合。
3.  **残差预测 (Residual Prediction)**：通过轻量级 MLP 预测点级 Logits。
4.  **最终推断**：$Logits_{final} = Logits_{superpoint} + \Delta Logits_{point}$。

## 2. 全局作用与协作 (System Role)

*   **承上**：
    *   **紧耦合**：直接消费 **Module 2 (CAFM)** 输出的 `enhanced_features_K` (增强超点) 和 `point_features_K` (原始点)。
    *   **无需重复计算**：**严禁**在此模块内重新运行 PointNet/MLP 提取点特征，必须复用 Module 2 的产出（算力节约关键点）。
*   **启下**：
    *   输出 `point_level_logits`。
    *   该输出将与 Ground Truth (原始点云标签) 计算 **Refinement Loss** (Cross Entropy)，这是提升边界精度的关键监督信号。

---

## 3. 输入输出接口定义 (I/O Interface)

请严格对齐 Module 2 的输出，保持维度一致。

### 3.1 初始化参数 (`__init__`)
*   `d_model` (int): 特征维度 (如 64)。
*   `num_classes` (int): 分类类别数 (如 S3DIS 为 13, ScanNet 为 20)。
*   `hidden_dim` (int): MLP 中间层维度，建议 `d_model` (KISS原则)。

### 3.2 前向传播输入 (`forward`)

| 变量名 | 形状 | 类型 | 来源/描述 |
| :--- | :--- | :--- | :--- |
| `enhanced_sp_features` | `[K, D]` | `float` | **Module 2 输出**。经过 Attention 增强后的超点特征。 |
| `point_features_K` | `[K, N, D]` | `float` | **Module 2 输出**。已编码的原始点特征。 |
| `coarse_logits` | `[K, C]` | `float` | **SPT Decoder**。该超点原本的预测 Logits (可选，用于残差连接)。 |

*   **注**：$K$=困难超点数, $N$=每个超点内的点数, $D$=特征维度, $C$=类别数。

### 3.3 输出 (Outputs)

| 变量名 | 形状 | 数据类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `point_logits` | `[K, N, C]` | `float` | 每个原始点的最终分类 Logits。用于计算细化 Loss 和生成最终掩码。 |

---

## 4. 核心算法逻辑 (Algorithms)

**步骤 1：维度扩展与对齐**
*   输入超点特征是 `[K, D]`。
*   需要扩展为 `[K, N, D]` 以便与点特征对应。
*   操作：`unsqueeze(1).expand(-1, N, -1)`。

**步骤 2：宏观-微观拼接**
*   将扩展后的超点特征与 `point_features_K` 在特征维度拼接。
*   拼接后形状：`[K, N, 2*D]`。
*   *物理意义*：每个点既知道“我在哪里（超点上下文）”，又知道“我是什么形状（点细节）”。

**步骤 3：残差 MLP 解码**
*   结构：`Linear(2D -> D)` -> `ReLU` -> `Linear(D -> C)`。
*   输出：$\Delta Logits$ (`[K, N, C]`)。

**步骤 4：残差加和 (Residual Addition)**
*   虽然 MLP 可以直接预测类别，但学习“残差”通常更容易收敛。
*   操作：`final_logits = coarse_logits.unsqueeze(1) + delta_logits`。
*   *注意*：`coarse_logits` 是全超点共享的，广播到 $N$ 个点上。

---

## 5. 给 AI Coder 的代码脚手架 (Scaffold)

这是最后一块拼图，代码逻辑简单直接，但维度操作需谨慎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualRefinementModule(nn.Module):
    """
    模块三：残差细化模块 (RRM)
    
    功能：
    接收超点特征和原始点特征，预测微观的点级分类 Logits，
    用于修正超点内部的边界分割错误。
    
    设计：
    Concat(Superpoint, Point) -> Shared MLP -> Residual Logits
    """
    
    def __init__(self, d_model: int = 64, num_classes: int = 13, hidden_dim: int = 64):
        super().__init__()
        
        # 输入维度是 2 * d_model (超点特征 + 点特征)
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self, 
        enhanced_sp_features: torch.Tensor, # [K, D] (来自 Mod 2)
        point_features_k: torch.Tensor,     # [K, N, D] (来自 Mod 2)
        coarse_logits: torch.Tensor         # [K, C] (来自 SPT Decoder)
    ) -> torch.Tensor:
        
        # --- 0. 防御性检查 ---
        if enhanced_sp_features.shape == 0:
            # 处理 K=0 的情况
            return torch.zeros(0, point_features_k.shape, coarse_logits.shape, 
                             device=enhanced_sp_features.device)
            
        K, N, D = point_features_k.shape
        
        # --- 1. 特征广播 (Broadcasting) ---
        # [K, D] -> [K, 1, D] -> [K, N, D]
        sp_expanded = enhanced_sp_features.unsqueeze(1).expand(-1, N, -1)
        
        # --- 2. 特征拼接 (Concatenation) ---
        # Cat([K, N, D], [K, N, D]) -> [K, N, 2D]
        combined_features = torch.cat([sp_expanded, point_features_k], dim=-1)
        
        # --- 3. 预测残差 Logits ---
        # [K, N, 2D] -> [K, N, C]
        delta_logits = self.refinement_head(combined_features)
        
        # --- 4. 残差连接 ---
        # Base: [K, C] -> [K, 1, C]
        base_logits = coarse_logits.unsqueeze(1)
        
        # Final = Base + Delta
        point_logits = base_logits + delta_logits
        
        return point_logits
```

---

## 6. 实现提示 (Implementation Notes)

请将以下重要指令传递给 AI Code Agent：

1.  **关于 Loss 计算 (关键协作)**：
    *   本模块**只负责前向传播**。
    *   请务必在注释中提醒用户：**Loss 计算不在本模块内**。用户需要在 `model.py` 的 `loss_function` 中，提取对应的 Ground Truth 点标签（需从 Dataloader 传入 `raw_point_labels` `[M, N]`），并只对 `hard_sp_indices` 对应的部分计算 CrossEntropyLoss。

2.  **显存优化**：
    *   尽管我们只处理 Top-K，但如果 $K$ 很大且 $N$ 很大（如 $N=1024$），`[K, N, 2D]` 的张量依然不小。
    *   **建议**：保持 $N$ 在合理范围（推荐 64 或 128）。如果 Dataloader 里的 $N$ 很大，建议在传入此模块前进行切片（例如只取前 128 个点）。

3.  **推理阶段 (Inference)**：
    *   在推理时，这些 `point_logits` 如何合并回最终的全景结果？
    *   **策略**：对于 Top-K 个超点，使用 `point_logits` 的 `argmax` 结果覆盖该区域的预测；对于其余超点，直接将超点的预测标签广播给内部所有点。

完成这一模块的设计，你的 **H-SPT** 核心链路（采样 -> 融合 -> 细化）就已形成闭环。