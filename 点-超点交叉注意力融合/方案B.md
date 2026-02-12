你好。基于我们确定的 **H-SPT** 架构与 **Design B** 路线，现在我们进入核心的 **“特征矫正”** 阶段。

**模块二：点-超点交叉注意力融合 (Point-Superpoint Cross-Attention Fusion)** 是整个系统的“心脏”。它连接了宏观（超点）与微观（原始点），负责将丢失的细节“注回”到模糊的超点特征中。

以下是为 **AI Code Agent** 准备的详细设计方案。

---

# 模块二需求规格说明书：点-超点交叉注意力融合 (PS-CAF)

**项目代号**：H-SPT (Hybrid-Superpoint Transformer)
**模块名称**：Module 2 - Point-Superpoint Cross-Attention Fusion (PS-CAF)
**设计范式**：Plugin Module (可插拔组件) / Attention Mechanism

## 1. 模块功能定义 (Functional Specification)

本模块的核心任务是**“特征矫正 (Feature Rectification)”**。
它接收模块一筛选出的“困难超点”，提取其内部的原始点云（几何+颜色），利用交叉注意力机制（Cross-Attention），让超点特征（Query）主动聚合原始点特征（Key/Value），从而恢复被池化操作丢失的细粒度信息。

**核心逻辑链**：
1.  **数据索引 (Gathering)**：根据 Module 1 提供的索引，提取对应的超点特征和内部原始点数据。
2.  **空间标准化 (Canonicalization)**：将原始点坐标转换为相对于超点中心的**局部坐标**，消除绝对位置影响，专注于局部几何形状。
3.  **微观编码 (Micro-Encoding)**：将原始点数据映射到高维特征空间。
4.  **交叉注意力 (Cross-Attention)**：$Q$ (超点) $\leftrightarrow$ $K,V$ (原始点) 交互。
5.  **残差融合 (Residual Fusion)**：将增强后的特征加回原超点特征。

## 2. 模块在全局的作用与协作 (System Role)

*   **承上**：直接消费 **Module 1 (AUS)** 输出的 `hard_sp_indices` 和 SPT Decoder 的 `sp_features`。
*   **启下**：输出 `enhanced_sp_features`。
    *   该特征将被送入 **Module 3 (Residual Refinement Head)** 进行亚超点级的掩码预测。
    *   同时，该特征会**替换**掉 SPT 主干网络中对应的原始特征，用于最终的语义分类 Loss 计算（Auxiliary Loss）。
*   **关键协作点**：必须确保特征维度 ($D_{model}$) 与 SPT 主干网络保持一致，以便无缝替换。

---

## 3. 输入输出接口定义 (I/O Interface)

请严格按照以下张量维度进行实现：

### 3.1 输入 (Inputs)

| 变量名 | 维度形状 | 数据类型 | 来源/描述 |
| :--- | :--- | :--- | :--- |
| `sp_features` | `[M, D]` | `float32` | SPT Decoder Level-1 的全部超点特征。 |
| `hard_sp_indices` | `[K]` | `int64` | **Module 1 输出**。本次需要处理的 Top-K 超点索引。 |
| `raw_points_idx` | `[M, N]` | `int64` | **Data Loader 提供**。预计算的索引表。每个超点对应 N 个原始点索引。 |
| `raw_coordinates` | `[Total_Pts, 3]` | `float32` | 整个场景/Batch 的原始点云坐标 $(x, y, z)$。 |
| `raw_colors` | `[Total_Pts, C_in]` | `float32` | 原始点云颜色 (RGB) 或强度，通常 $C_{in}=3$ 或 $4$。 |
| `sp_centroids` | `[M, 3]` | `float32` | SPT 预计算的超点质心坐标，用于去绝对坐标化。 |

*   **注**：$M$=超点总数, $K$=困难超点数, $N$=每个超点采样的原始点数 (如 128), $D$=特征维度 (如 64), $Total\_Pts$=原始点总数。

### 3.2 输出 (Outputs)

| 变量名 | 维度形状 | 数据类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `enhanced_sp_features` | `[K, D]` | `float32` | 仅包含被增强的那 K 个超点的特征。 |
| `fused_global_features` | `[M, D]` | `float32` | (可选便捷输出) 已将增强特征写回原位置的完整特征图，方便直接传给 Loss。 |

---

## 4. 技术栈与核心算法 (Tech Stack & Algorithms)

**技术栈**：PyTorch (`torch.nn.MultiheadAttention` 或 手写 Scaled Dot-Product Attention)

**算法流程详情**：

### 步骤 1：数据准备 (Data Gathering) & 坐标标准化
这是本模块最容易出错的地方。不能直接用绝对坐标学习，否则网络记不住“形状”。

*   **索引提取**：
    *   利用 `hard_sp_indices` 从 `sp_features` 提取 $Q_{raw}$: `[K, D]`。
    *   利用 `hard_sp_indices` 从 `raw_points_idx` 提取相关点索引: `[K, N]`。
    *   利用点索引从 `raw_coordinates` 和 `raw_colors` 提取数据: `batch_pos` `[K, N, 3]`, `batch_feat` `[K, N, 3]`。
*   **标准化 (Canonicalization)**:
    *   提取对应超点的质心: `centroids` `[K, 3]`。
    *   **关键公式**: `local_pos = batch_pos - centroids.unsqueeze(1)`。
    *   *目的*：让网络理解“这个点在超点的左上角”，而不是“这个点在世界坐标 (100, 200, 50)”。

### 步骤 2：微观特征编码 (Micro-Point Encoder)
我们需要把原始点的低维数据 (3+3=6) 升维到与超点一致的 $D$ 维度。

*   **网络结构**: Shared MLP (1x1 Conv)。
    *   Input: `Concat(local_pos, batch_feat)` $\to$ `[K, N, 6]`
    *   Layers: `Linear(6 -> 32) -> ReLU -> Linear(32 -> D)`
    *   Output: $K_{raw}, V_{raw}$ $\to$ `[K, N, D]`

### 3. 交叉注意力 (Cross-Attention)
*   **Query**: 超点特征 $Q = \text{Linear}(Q_{raw}) \to [K, 1, D]$ (注意这里 Sequence Length 为 1)。
*   **Key/Value**: 原始点特征 $K = V = \text{PointEncoderOutput} \to [K, N, D]$。
*   **Attention**:
    $$ \text{Attn} = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{D}}\right) \cdot V $$
    *   维度变化: `[K, 1, N] @ [K, N, D] -> [K, 1, D]`。
    *   *物理意义*: 超点根据自身的特征需求，动态加权聚合内部的 128 个点。

### 4. 残差连接与前馈 (FFN)
*   `out = Dropout(Attn_Output) + Q_raw` (残差连接)
*   `out = LayerNorm(out)`
*   (可选) 增加一个 FFN: `out = FFN(out) + out`

---

## 5. 实现提示与检查清单 (Implementation Notes)

请将以下指令传递给 Code Agent：

1.  **处理 K=0 的情况**：
    *   **防御性编程**：Module 1 可能没选出任何点（虽然设有 min=1，但需防万一）。如果 `hard_sp_indices` 为空，直接返回原始 `sp_features`，跳过计算。

2.  **维度对齐陷阱**：
    *   PyTorch 官方的 `MultiheadAttention` 默认输入形状是 `(Seq_Len, Batch, Dim)`。
    *   而在点云任务中，我们通常持有 `(Batch, N, Dim)`。
    *   **指令**：明确要求使用 `batch_first=True` 参数，或者在输入前手动 `permute`。鉴于这里 $Q$ 的长度固定为 1，建议直接手写简单的 `Scaled Dot-Product Attention`，代码更轻量且易于控制。

3.  **内存优化**：
    *   `raw_points_idx` 表可能很大。确保只在 `forward` 阶段进行切片索引 (`index_select` 或 `gather`)，不要复制整个大表。

4.  **初始化**：
    *   Attention 最后的输出投影层建议初始化为接近 0 的权重，或使用 `LayerScale`，使得训练初期模块近似恒等映射（Identity），保证 SPT 原有性能不崩塌。

---

**给 AI Code Agent 的指令 (Prompt) 示例**：

> "请基于上述规格说明书，编写 `CrossAttentionFusion` 类。
>
> **关键要求**：
> 1. 输入包含 `raw_points_idx` 映射表，你需要实现从全局 `raw_coordinates` 取点的逻辑。
> 2. 必须实现**坐标局部化 (Canonicalization)**：原始点坐标减去超点质心。
> 3. 手写一个轻量级的 Attention 逻辑（不要用 nn.MultiheadAttention，太重了），Query 是超点，Key/Value 是原始点。
> 4. 加入 Residual Connection 和 LayerNorm。
> 5. 输出需要包含更新后的全局特征图 `fused_global_features`（未被选中的位置保持原样，被选中的位置更新）。"