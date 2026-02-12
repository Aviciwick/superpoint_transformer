这是一份专为 AI Coding Agent（如 GitHub Copilot, Cursor 或专门的工程师）准备的**详细技术实施文档**。

你可以直接将以下内容复制并发送给 AI Coding Agent，它将能够毫无歧义地生成高质量的代码。

---

## 任务：实现 H-SPT 模型的“模块一：自适应不确定性采样器”

### 1. 模块概述 (Module Overview)
本任务要求在现有的 **Superpoint Transformer (SPT)** 架构中，实现一个名为 `AdaptiveUncertaintySampler` 的 PyTorch 模块。

*   **核心功能**：该模块充当“过滤器”。它接收所有超点的初步分类预测和几何特征，计算每个超点的“难度得分”，并筛选出 **Top-K%** 最难的超点（Hard Superpoints）。
*   **算法策略**：采用 **“方案 B：熵 + 规则”**。即利用预测的不确定性（熵）和几何的复杂性（规则特征）进行加权打分，无需额外的可学习参数。

### 2. 全局作用 (Global Role)
在整个 **H-SPT (Hybrid-SPT)** 架构中，该模块位于 **Backbone Decoder** 与 **Refinement Heads** 之间：
1.  **上游 (Upstream)**：接收 Backbone (SPT) Level-1 的输出特征和 Logits。
2.  **下游 (Downstream)**：向后续的“交叉注意力模块”和“残差细化头”提供 **索引列表 (Indices)**。
3.  **价值**：实现计算资源的动态分配。仅针对选中的“困难超点”进行昂贵的点级交互，从而在提升精度的同时控制显存占用。

### 3. 输入输出定义 (I/O Specification)

#### 输入 (Inputs)
该模块的 `forward` 函数需要接收以下 Tensor：

1.  **`coarse_logits`**
    *   **形状**：`[N_total, Num_Classes]`
    *   **描述**：SPT Decoder 对 Level-1 超点输出的原始分类分数（未经过 Softmax）。
    *   **注意**：`N_total` 是当前 Stacked Batch 中所有超点的总数。

2.  **`handcrafted_features`**
    *   **形状**：`[N_total, D_feat]`
    *   **描述**：预处理阶段计算好的几何/辐射特征。
    *   **关键维度**：需要从 `D_feat` 中提取特定的几何特征（如 **Scattering/散射度** 或 **Verticality/垂直度**）。

#### 参数配置 (Configuration)
初始化模块时需要的超参数：
*   `sample_ratio` (float): 采样比例，例如 `0.2` 表示选取前 20%。
*   `alpha` (float): 语义熵的权重系数，默认 `1.0`。
*   `beta` (float): 几何特征的权重系数，默认 `0.5`。
*   `geo_feature_index` (int): 几何特征在 `handcrafted_features` 张量中的通道索引（需根据数据集定义确定）。

#### 输出 (Outputs)
1.  **`hard_indices`**
    *   **形状**：`[K]`，其中 $K = \text{ceil}(N_{total} \times \text{sample\_ratio})$。
    *   **描述**：被选中的超点在 `N_total` 维度上的**绝对索引** (LongTensor)。
    *   **用途**：后续模块将使用 `torch.index_select` 或 `gather` 根据此索引提取特征。

---

### 4. 算法逻辑与技术栈 (Algorithms & Tech Stack)

**技术栈**：Python, PyTorch (Vectorized Operations ONLY, **No Loops**)

#### 步骤 1：计算语义不确定性 (Semantic Uncertainty)
使用 **香农熵 (Shannon Entropy)** 衡量分类的困惑度。
*   **算法**：
    $$ P = \text{Softmax}(\text{logits}) $$
    $$ H = -\sum (P \times \log(P + \epsilon)) $$
*   **归一化**：建议将 $H$ 除以 $\log(\text{Num\_Classes})$，使其范围归一化到 $[0, 1]$。
*   **数值稳定性**：在 `log` 中必须加入 $\epsilon = 1e^{-6}$ 防止 NaN。

#### 步骤 2：获取几何显著性 (Geometric Saliency)
使用预计算的特征衡量物理结构的复杂性。
*   **算法**：
    $$ G = \text{handcrafted\_features}[:, \text{geo\_feature\_index}] $$
*   **数据清洗**：
    *   虽然 SPT 预处理通常已归一化，但为了鲁棒性，建议在代码中强制将 $G$ 裁剪或归一化到 $[0, 1]$ 范围。
    *   如果输入特征包含 NaN，需填充为 0。

#### 步骤 3：加权融合与排序 (Fusion & Ranking)
*   **算法**：
    $$ \text{Score} = \alpha \cdot H + \beta \cdot G $$
*   **排序**：
    *   使用 `torch.topk(Score, k)` 获取最高分的 $K$ 个元素的索引。
    *   **Batch 处理策略**：直接对整个 Stacked Batch 进行全局排序（Global Top-K）。这能自动将计算量分配给场景中更复杂的区域，无需按 Batch ID 循环。

---

### 5. 给 AI Coder 的具体实现提示 (Implementation Hints)

请按以下要求编写代码：

1.  **类结构**：继承自 `nn.Module`。
2.  **无状态**：该方案 (Plan B) 不需要训练参数（Parameters），不需要 `requires_grad=True`。
3.  **安全性**：
    *   如果 `handcrafted_features` 为 `None`（某些消融实验可能不提供），则自动回退到仅使用熵（`beta=0`）。
    *   确保 `k` 至少为 1，防止后续代码崩溃。
4.  **性能**：所有计算必须在 GPU 上并行完成，严禁使用 Python `for` 循环遍历节点。

### 6. 代码脚手架 (Scaffold)

```python
import torch
import torch.nn as nn

class AdaptiveUncertaintySampler(nn.Module):
    def __init__(self, sample_ratio=0.2, alpha=1.0, beta=0.5, geo_feat_idx=None):
        super().__init__()
        # TODO: Initialize hyperparameters
        pass

    def forward(self, coarse_logits, handcrafted_features=None):
        """
        Inputs:
            coarse_logits: [N, C]
            handcrafted_features: [N, D] (Optional)
        Returns:
            hard_indices: [K]
        """
        # TODO: 1. Calculate Softmax & Entropy
        # TODO: 2. Extract Geometric Feature (if available)
        # TODO: 3. Compute Weighted Score
        # TODO: 4. Perform Top-K Selection
        pass
```