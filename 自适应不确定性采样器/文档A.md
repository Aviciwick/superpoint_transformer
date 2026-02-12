这是一份专为**AI 代码助手 (AI Code Agent)** 准备的工程级需求规格说明书。该文档严格遵循我们确定的 **Design B（熵 + 规则）** 路线，基于 **KISS 原则**，剔除了所有复杂的训练组件（MLP、Auxiliary Loss），确保模块轻量、可控且即插即用。

你可以将以下内容直接作为 Prompt 发送给负责编码的 AI Agent。

---

# 模块一需求规格说明书：自适应不确定性采样器 (Rule-Based AUS)

**项目代号**：H-SPT (Hybrid-Superpoint Transformer)
**模块名称**：Module 1 - Adaptive Uncertainty Sampler (AUS)
**设计范式**：Heuristic / Rule-Based (Non-learnable)

## 1. 模块功能定义 (Functional Specification)

本模块是一个**确定性（Deterministic）的过滤器**。其核心功能是计算每个超点（Superpoint）的“不确定性得分”，并根据得分筛选出 Top-K% 的困难超点。

具体包含三个原子操作：
1.  **语义评估**：基于分类器输出的 Logits，计算香农熵（Shannon Entropy），衡量模型对该超点语义识别的困惑度。
2.  **几何评估**：从预计算的手工特征中提取“散射度（Scattering）”或“垂直度”，衡量该超点内部的物理几何复杂性。
3.  **Top-K 排序与索引**：将上述两者加权归一化后，在全局（或 Batch 范围内）选出得分最高的 $K$ 个超点，并返回其索引。

## 2. 全局作用与架构位置 (System Role)

*   **角色**：整个 H-SPT 系统的**“守门员” (Gatekeeper)**。
*   **架构位置**：位于 SPT 模型 `Decoder` 的末端，在输出最终全景分割结果之前。
*   **上下游关系**：
    *   **上游**：接收 Decoder 产生的 Level-1 超点特征和初步分类预测。
    *   **下游**：向 **模块二 (Cross-Attention)** 和 **模块三 (Refinement Head)** 提供一份“困难超点名单（索引）”。
*   **价值**：解决“算力预算”问题。通过仅筛选前 20% 的困难区域进入后续昂贵的点级交互模块，避免显存爆炸，同时保证精细化处理聚焦在物体边界和复杂结构上。

## 3. 输入输出接口定义 (I/O Interface)

请严格按照以下张量维度进行实现：

### 3.1 输入 (Inputs)
| 变量名 | 维度形状 | 数据类型 | 来源/描述 |
| :--- | :--- | :--- | :--- |
| `coarse_logits` | `[M, C]` | `torch.float32` | SPT Decoder Level-1 的原始输出 (未过 Softmax)。<br>$M$: 当前 Batch 超点总数, $C$: 类别数。 |
| `handcrafted_features` | `[M, D_feat]` | `torch.float32` | 数据加载器中预计算的几何特征。<br>需包含 Scattering/Planarity 等通道。 |
| `batch_indices` | `[M]` | `torch.int64` | (可选) 标记每个超点属于哪个 Batch 样本，用于分场景 Top-K。**MVP版本可忽略，直接全局排序**。 |

### 3.2 超参数 (Hyperparameters)
*   `topk_ratio` (float): 采样比例，默认 `0.20` (即 20%)。
*   `alpha` (float): 语义熵权重，默认 `0.7`。
*   `beta` (float): 几何特征权重，默认 `0.3`。
*   `scatter_idx` (int): `handcrafted_features` 中对应散射度的通道索引（需核对 Data Loader）。

### 3.3 输出 (Outputs)
| 变量名 | 维度形状 | 数据类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `hard_sp_indices` | `[K]` | `torch.int64` | 选中的 Top-K 个超点在输入 Tensor (`dim=0`) 中的绝对索引。 |
| `debug_scores` | `[M]` | `torch.float32` | (可选) 所有超点的评分，用于可视化调试。 |

---

## 4. 技术栈与核心算法 (Tech Stack & Algorithms)

**技术栈**：PyTorch (Tensor Operations), `torch.distributions`

**核心算法逻辑**：

1.  **标准化语义熵计算**：
    *   先对 Logits 做 Softmax 得到概率 $P$。
    *   计算熵 $H = -\sum P \log P$。
    *   **关键归一化**：$H_{norm} = H / \log(C)$，确保值域在 $$。

2.  **鲁棒的几何特征归一化**：
    *   直接读取 `handcrafted_features[:, scatter_idx]`。
    *   **关键操作**：必须在当前 Batch 内进行 **Min-Max Normalization**。
    *   公式：$G_{norm} = \frac{G - G_{min}}{G_{max} - G_{min} + \epsilon}$。
    *   *理由*：不同场景的几何特征数值波动极大，不归一化会导致几何项淹没语义项或失效。

3.  **加权评分**：
    *   $S_{total} = \alpha \cdot H_{norm} + \beta \cdot G_{norm}$。

4.  **Top-K 采样**：
    *   使用 `torch.topk(S_total, k=K)`。
    *   $K = \text{max}(1, \lfloor M \times \text{topk\_ratio} \rfloor)$，防止 $K=0$ 导致崩溃。

---

## 5. 实现提示与检查清单 (Implementation Notes)

请 Code Agent 在编写代码时特别关注以下“隐形”工程细节：

1.  **Dataloader 索引确认**：
    *   必须在代码注释中明确：需要用户核对 `dataset.py`，确认 `handcrafted_features` 张量中，哪个维度是 **Scattering (散射度)**。通常 SPT 顺序为 `[linearity, planarity, scattering, verticality]`，则索引为 `2`。

2.  **数据流透传 (Data Engineering)**：
    *   这是本模块最关键的**外部依赖**。虽然本模块只输出超点索引，但后续模块需要知道这些超点内部的原始点（Raw Points）。
    *   **要求**：请提示用户，需要在 Dataset 的 `__getitem__` 中增加一个预索引步骤，构建一个 `sp_to_raw_indices` 映射表（Tensor），以便后续通过 `hard_sp_indices` 直接查表取点，而不是在 GPU 上做极其低效的搜索。

3.  **无需梯度**：
    *   本模块所有操作均不涉及参数学习。建议使用 `torch.no_grad()` 上下文或确保操作不打断计算图（虽然本模块本身无参数，但不应影响 `coarse_logits` 的梯度回传）。**注意：** 如果后续要用分类 Loss，Logits 的梯度需要正常回传，但采样逻辑本身不需要梯度。

4.  **文件结构建议**：
    *   建议新建 `models/sampler.py` 封装此类，保持 `model.py` 清洁。

---

**给 AI Code Agent 的指令 (Prompt)**：
> "请基于上述规格说明书，使用 PyTorch 编写 `RuleBasedSampler` 类。代码需包含完整的类型注解、鲁棒的除零保护（Epsilon），以及详细的输入输出文档字符串。同时，请给出一小段集成到 SPT `forward` 函数中的示例代码。"