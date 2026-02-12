这是**模块三：残差细化头 (Residual Refinement Head, RRH)** 的详细设计方案。

作为 H-SPT 系统的**“尾部”**，它是实现“亚超点级（Sub-superpoint）精度”的执行者。它直接消费模块二（PS-CAF）增强后的特征，打破超点的物理边界限制，对内部像素进行点级分类。

以下是为 AI Code Agent 准备的标准化技术文档。

---

## 模块三需求规格说明书：残差细化头 (RRH)

**项目代号**：H-SPT (Hybrid-Superpoint Transformer)
**模块名称**：Module 3 - Residual Refinement Head (RRH)
**设计范式**：Point-wise Classifier / Dense Prediction Head

### 1. 模块功能定义 (Functional Specification)
本模块的核心任务是**“边界重构 (Boundary Reconstruction)”**。
它不再将超点视为一个不可分割的原子，而是深入其内部，根据模块二提供的“宏观上下文（超点特征）”和“微观细节（点特征）”，为采样区域内的每一个原始点预测一个语义类别（或类别偏移量）。

**核心逻辑链**：
1.  **上下文广播 (Context Broadcasting)**：将模块二增强后的超点特征（$1 \times D$）复制广播，使其与内部 $N$ 个原始点一一对应。
2.  **特征融合 (Feature Concatenation)**：将广播后的宏观特征与微观点特征（来自模块二）拼接。
    *   *物理含义*：“我是这个超点的一部分（宏观），我长这个样子（微观）”。
3.  **残差预测 (Residual/Dense Prediction)**：通过 MLP 预测每个点的语义 Logits。
4.  **训练/推理分支**：
    *   **训练时**：直接计算点级 Cross-Entropy Loss（监督信号来自原始点云 GT）。
    *   **推理时**：生成点级掩码，修正超点的“锯齿边缘”，甚至将超点分裂为两个不同的物体。

### 2. 模块在全局的作用与协作 (System Role)
*   **承上**：
    *   **强依赖**模块二 (PS-CAF) 的输出：`enhanced_sp_features` (宏观) 和 `point_features_K` (微观)。**这避免了重复计算点特征，极大节省算力。**
    *   **依赖**模块一 (AUS) 的 `hard_sp_indices` (虽然模块二已经过滤过数据，但 RRH 可能需要索引来溯源)。
*   **启下**：
    *   输出 `point_logits_K`。
    *   在 Loss 计算阶段，这是 $L_{refine}$ 的来源。
    *   在 SuperCluster 图聚类阶段，它提供了一个“点级修正补丁”。

---

### 3. 输入输出接口定义 (I/O Interface)

请严格对齐模块二的输出：

#### 3.1 输入 (Inputs)
| 变量名 | 维度形状 | 数据类型 | 来源 |
| :--- | :--- | :--- | :--- |
| **`enhanced_sp_features`** | `[K, D]` | float32 | **模块二输出**。已融合了细节信息的超点特征。 |
| **`point_features_K`** | `[K, N, D]` | float32 | **模块二输出**。超点内部 $N$ 个点的独立特征（已编码）。 |
| **`sp_logits`** | `[K, C]` | float32 | (可选) SPT Decoder 对这些超点的原始分类预测。用于残差加法。 |

*   *注*：$K$ = 困难超点数, $N$ = 每个超点的采样点数, $D$ = 特征维度, $C$ = 类别数。

#### 3.2 输出 (Outputs)
| 变量名 | 维度形状 | 数据类型 | 描述 |
| :--- | :--- | :--- | :--- |
| **`refinement_logits`** | `[K, N, C]` | float32 | **每个原始点**的语义分类 Logits。 |

---

### 4. 技术栈与核心算法 (Tech Stack & Algorithms)

**技术栈**：PyTorch (`torch.cat`, `torch.expand`)

#### 算法流程详情：

**步骤 1：维度对齐与广播 (Broadcasting)**
*   输入 `enhanced_sp_features` 形状为 `[K, D]`。
*   我们需要将其扩展为 `[K, N, D]` 以便与点特征对齐。
*   操作：`sp_expanded = sp_features.unsqueeze(1).expand(-1, N, -1)`。

**步骤 2：特征拼接 (Concatenation)**
*   将宏观特征与微观特征在通道维度拼接。
*   `fused_input = cat([sp_expanded, point_features_K], dim=-1)`
*   结果形状：`[K, N, 2*D]`。

**步骤 3：密集预测 (Dense Prediction MLP)**
*   使用一个多层感知机将特征映射到类别空间。
*   结构建议：`Linear(2D -> D) -> ReLU -> Dropout -> Linear(D -> C)`。
*   *可选的高级残差设计*：
    *   如果传入了 `sp_logits` (`[K, C]`)，可以将其广播为 `[K, N, C]`。
    *   最终输出 = `MLP_Output + sp_logits_expanded`。
    *   这强调了模型只需要学习“当前点与所属超点的类别偏差”，通常收敛更快。

---

### 5. 实现提示与检查清单 (Implementation Notes)

请将以下指令传递给 Code Agent：

1.  **处理 K=0 的情况**：
    *   防御性编程：如果模块一没选出任何点（`K=0`），模块二会传空 Tensor 过来。
    *   指令：检测 `enhanced_sp_features.size(0) == 0`，如果是，直接返回空的 Tensor（形状 `[0, N, C]`），确保后续 Loss 计算代码不报错。

2.  **显存优化**：
    *   虽然拼接后维度是 `2D`，但由于只涉及 $K$ 个超点（$K \approx 20\% M$），显存压力通常可控。
    *   指令：尽量使用 `inplace` 的 `ReLU`。

3.  **残差链接的选择**：
    *   为了保持通用性，建议实现**Logits 残差**（如果提供了父超点 Logits）或 **纯预测**（如果没有提供）。代码中可以通过 `if sp_logits is not None` 来分支处理。

---

### 6. 给 AI Code Agent 的指令 (Prompt) 示例

```text
请基于上述《模块三需求规格说明书 (RRH)》，编写 ResidualRefinementHead 类。

关键技术约束如下：

1. 输入对齐：
   forward 函数必须接收 enhanced_sp_features [K, D] 和 point_features_K [K, N, D] 这两个来自 Module 2 的输出。
   可选接收 sp_logits [K, C] 用于残差加法。

2. 广播逻辑：
   必须正确使用 unsqueeze(1) 和 expand(-1, N, -1) 将超点特征广播到与点特征相同的形状。

3. 网络结构：
   使用一个轻量级的 MLP：Linear(2*D -> D) -> ReLU -> Dropout -> Linear(D -> Num_Classes)。
   注意输入维度是 2*D (拼接了超点特征和点特征)。

4. 残差预测 (Residual Logic)：
   如果输入中提供了 sp_logits，请执行 output = mlp_output + sp_logits.unsqueeze(1)；
   否则直接返回 output = mlp_output。
   这允许模型只学习“点相对于超点”的类别偏差。

5. 边界情况：
   如果输入 K=0，请立即返回空 Tensor，不要运行 MLP，防止 Shape 错误。
```