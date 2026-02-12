这是一个为 AI Coding Agent 准备的**模块二：交叉注意力融合模块 (Cross-Attention Fusion Module, CAFM)** 的详细设计方案。

此设计文档与上一条“自适应采样器”的代码完全对齐，并为下一条“残差细化头”做好了数据准备。

---

## 任务：实现 H-SPT 模型的“模块二：交叉注意力融合模块”

### 1. 模块概述 (Module Overview)
本模块是 H-SPT 的**特征增强器**。它仅对模块一筛选出的“困难超点”进行处理，通过 **Cross-Attention** 机制，让粗粒度的超点特征（Query）主动“吸收”其内部原始点云（Key/Value）的细粒度几何与纹理信息。

*   **设计范式**：In-Loop Feature Enhancement (Coarse-to-Fine)
*   **架构位置**：
    *   **上游**：模块一 (Adaptive Uncertainty Sampler)。
    *   **下游**：模块三 (Residual Refinement Head) 及 SPT 的最终分类层。
*   **核心价值**：解决“量化误差”导致的特征模糊问题，使超点特征具备微观感知能力。

---

### 2. 模块协作与数据对齐 (Module Alignment)

为了保证流水线顺畅，本模块与上下文的交互如下：

*   **与模块一的对齐**：
    *   **输入**：直接接收模块一输出的 `hard_sp_indices` (Top-K 索引)。
    *   **逻辑**：只处理这些索引对应的超点，其他超点特征保持不变（或不参与计算）。

*   **与 Dataset/Dataloader 的对齐 (关键依赖)**：
    *   **前置条件**：为了 GPU 高效并行，我们**强烈建议**在 Dataloader 阶段完成“点云归组”。
    *   **数据格式**：模块期望接收一个预处理好的张量 `sp_raw_points`，形状为 `[Total_SP, N_sample, D_raw]`。
    *   *解释*：每个超点预先采样 $N$ 个内部点（如 64 点）。不足 $N$ 点的 padding，超过的 downsample。这避免了在 `forward` 中进行复杂的动态索引查找。

*   **与模块三的对齐**：
    *   **输出**：不仅输出增强后的 `enhanced_sp_features`，还要顺便输出计算中间产生的 `projected_point_features`（投影后的点特征）。
    *   *原因*：模块三也需要点特征来做掩码预测。在这里算好传过去，避免重复计算，节省算力。

---

### 3. 输入输出定义 (I/O Specification)

#### 初始化参数 (`__init__`)
*   `d_model` (int): 超点特征维度（与 SPT Decoder 输出一致，如 64 或 128）。
*   `d_raw` (int): 原始点云维度（通常 3+3=6，即 xyz+rgb，或者 3+feature）。
*   `n_heads` (int): Multi-head Attention 头数，默认 4。
*   `dropout` (float): 默认 0.1。

#### 输入 (`forward`)
1.  **`hard_sp_indices`**: `[K]` (LongTensor) - 来自模块一。
2.  **`all_sp_features`**: `[M, d_model]` - SPT Decoder 输出的所有超点特征。
3.  **`all_sp_centroids`**: `[M, 3]` - 所有超点的物理中心坐标（用于局部坐标归一化）。
4.  **`packed_raw_points`**: `[M, N_sample, d_raw]` - 预处理好的原始点数据。
    *   前 3 维必须是 $(x, y, z)$ 绝对坐标。

#### 输出 (`forward` returns)
1.  **`enhanced_features_K`**: `[K, d_model]` - 仅包含被增强的 K 个超点特征。
2.  **`point_features_K`**: `[K, N_sample, d_model]` - 被选中超点内部的、经过 MLP 编码的点特征（供模块三复用）。

---

### 4. 算法逻辑与技术栈 (Algorithms & Tech Stack)

**技术栈**: PyTorch, `torch.nn.MultiheadAttention` (或手动实现 Attention 以优化显存).

#### 步骤 1：数据收集 (Gathering)
使用 `hard_sp_indices` 从全局数据中提取 Top-K 数据。
*   `K_sp_feat = all_sp_features[indices]` -> `[K, d_model]`
*   `K_centroids = all_sp_centroids[indices]` -> `[K, 3]`
*   `K_raw_points = packed_raw_points[indices]` -> `[K, N, d_raw]`

#### 步骤 2：局部坐标编码 (Local Geometry Encoding)
**这是让模型理解“形状”而非“位置”的关键。**
1.  **坐标解耦**：从 `K_raw_points` 分离出 XYZ `[K, N, 3]` 和 属性 `[K, N, D_attr]`。
2.  **归一化**：$XYZ_{local} = XYZ_{global} - Centroid_{expanded}$。
3.  **特征拼接**：$Input_{pts} = \text{Concat}(XYZ_{local}, \text{Attributes})$。

#### 步骤 3：点云特征嵌入 (Point Embedding)
使用 Shared MLP (类似 PointNet) 将点云映射到高维。
*   Network: `Linear(d_raw) -> ReLU -> Linear(d_model)`.
*   Result: `F_pts` (`Key/Value Source`) shape `[K, N, d_model]`.

#### 步骤 4：交叉注意力 (Cross-Attention)
*   **Query**: `K_sp_feat` (unsqueeze dim 1 -> `[K, 1, d_model]`).
*   **Key/Value**: `F_pts` (`[K, N, d_model]`).
*   **Positional Encoding**: 可选，通常 $XYZ_{local}$ 已经隐含了位置信息，可跳过显式 PE 以简化设计。
*   **Attention Operation**:
    $$ \text{Attn} = \text{Softmax}(\frac{Q \cdot K^T}{\sqrt{d}}) \cdot V $$
    *   *实现提示*：可以使用 PyTorch 的 `MultiheadAttention`，注意 `batch_first=True`。

#### 步骤 5：残差融合 (Residual Fusion)
*   $$ F_{out} = \text{Norm}(Q + \text{Dropout}(\text{Attn\_Output})) $$
*   使用 `LayerNorm`。

---

### 5. 给 AI Coder 的具体实现提示

请按以下要求编写代码：
1.  **维度检查**：在 `forward` 开头对输入形状进行 assert，确保 `packed_raw_points` 的维度正确。
2.  **KISS 原则**：不要实现完整的 PointNet++ Set Abstraction，只用简单的 MLP 做点编码即可。
3.  **计算图**：确保梯度能通过 Attention 回传到 Point Encoder，同时也回传到 `all_sp_features`。
4.  **返回类型**：返回一个 `NamedTuple` 或字典，以便后续扩展，明确区分“增强后的特征”和“点特征”。

---

### 6. 代码脚手架 (Scaffold)

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict

class CrossAttentionFusionModule(nn.Module):
    def __init__(
        self, 
        d_model: int = 64, 
        d_raw: int = 6, 
        n_heads: int = 4, 
        dropout: float = 0.1,
        n_sample: int = 64
    ):
        """
        Args:
            d_model: 超点特征维度 (SPT hidden dim)
            d_raw: 原始点云维度 (xyz + rgb/intensity)
            n_heads: Attention 头数
            n_sample: 每个超点内部采样的点数 (用于 shape check)
        """
        super().__init__()
        self.d_model = d_model
        self.n_sample = n_sample
        
        # 1. Point Encoder (Mini-PointNet)
        # 将局部坐标+颜色映射到与超点特征相同的维度
        self.point_encoder = nn.Sequential(
            nn.Linear(d_raw, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Cross-Attention
        # batch_first=True: [Batch, Seq_Len, Dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Residual Connection & Norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 可选: Feed Forward Network (FFN) 进一步增强
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        hard_sp_indices: torch.Tensor,
        all_sp_features: torch.Tensor,
        all_sp_centroids: torch.Tensor,
        packed_raw_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hard_sp_indices: [K] - 选中的超点索引
            all_sp_features: [M, D] - 所有超点特征
            all_sp_centroids: [M, 3] - 所有超点中心
            packed_raw_points: [M, N_sample, D_raw] - 所有超点的原始点数据 (xyz在前3维)
            
        Returns:
            enhanced_features_K: [K, D] - 增强后的 K 个超点特征
            point_features_K: [K, N_sample, D] - 编码后的点特征 (供 Module 3 使用)
        """
        # ===========================
        # Step 1: Gather Data (Top-K)
        # ===========================
        # 提取 K 个超点的数据
        # indices shape: [K]
        # gathered_sp_feat: [K, D]
        gathered_sp_feat = all_sp_features[hard_sp_indices]
        
        # gathered_centroids: [K, 3]
        gathered_centroids = all_sp_centroids[hard_sp_indices]
        
        # gathered_raw_points: [K, N, D_raw]
        gathered_raw_points = packed_raw_points[hard_sp_indices]
        
        # =======================================
        # Step 2: Local Encoding (Pre-Attn Prep)
        # =======================================
        # 分离 XYZ 和 属性
        raw_xyz = gathered_raw_points[..., :3]  # [K, N, 3]
        raw_attr = gathered_raw_points[..., 3:] # [K, N, D_raw-3]
        
        # 计算局部相对坐标 (Local Coordinates)
        # centroid: [K, 3] -> [K, 1, 3]
        local_xyz = raw_xyz - gathered_centroids.unsqueeze(1)
        
        # 重新拼接: Local XYZ + Attributes
        point_input = torch.cat([local_xyz, raw_attr], dim=-1) # [K, N, D_raw]
        
        # 映射到特征空间
        # K_V_src: [K, N, D_model]
        point_features_K = self.point_encoder(point_input)
        
        # Query: 超点特征 [K, 1, D_model]
        Q_src = gathered_sp_feat.unsqueeze(1)
        
        # ===========================
        # Step 3: Cross-Attention
        # ===========================
        # Query: Superpoint (Macro)
        # Key/Value: Points (Micro)
        attn_out, _ = self.cross_attn(
            query=Q_src,
            key=point_features_K,
            value=point_features_K
        )
        # attn_out: [K, 1, D_model]
        
        # ===========================
        # Step 4: Residual & Norm
        # ===========================
        # 残差连接 1: Attention Result + Original SP Feature
        x = gathered_sp_feat + self.dropout(attn_out.squeeze(1))
        x = self.norm(x)
        
        # 残差连接 2: FFN (Optional but recommended)
        x2 = self.ffn(x)
        enhanced_features_K = self.norm_ffn(x + x2)
        
        return enhanced_features_K, point_features_K
```