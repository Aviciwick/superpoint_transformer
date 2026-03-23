# 推理性能优化计划

## 目标
优化 `batch_inference.py` 的推理性能，将显卡利用率从 ~30% 提升到 ~70-80%。

## 问题分析

### 当前瓶颈
1. **串行数据加载**：直接使用 `dataset[idx]` 从磁盘读取，GPU 空闲等待
2. **无预取机制**：每次推理完成后才开始加载下一个场景
3. **同步保存**：主线程阻塞保存结果，GPU 空闲等待

### 时间分布
| 操作 | 当前占比 | GPU 状态 |
|------|---------|---------|
| 数据加载 | ~30% | 空闲等待 |
| CPU→GPU 传输 | ~10% | 部分工作 |
| GPU 推理 | ~25% | 真正工作 |
| GPU→CPU 同步 | ~10% | 等待同步 |
| 文件保存 | ~25% | 空闲等待 |

## 优化方案

### 方案一：使用 DataLoader（推荐，高优先级）

**修改内容：**
1. 引入 `torch.utils.data.DataLoader`
2. 使用项目已有的 `NAGBatch.collate` 函数
3. 配置多进程加载参数

**代码修改位置：** `batch_inference.py` 第 356-446 行

**具体步骤：**
1. 导入必要的模块
2. 创建 DataLoader 实例
3. 修改推理循环使用 DataLoader
4. 处理 NAGBatch 对象

**预期效果：** GPU 利用率提升到 ~70%

### 方案二：异步保存（可选，中优先级）

**修改内容：**
1. 使用线程池异步保存结果
2. 推理和保存并行执行

**预期效果：** 额外提升 ~10% 效率

## 实施步骤

### Step 1: 添加 DataLoader 参数
- 在 `main()` 函数中添加 `--num_workers` 参数
- 添加 `--pin_memory` 参数
- 添加 `--prefetch_factor` 参数

### Step 2: 创建 DataLoader
- 替换 `for idx in scene_indices` 循环
- 使用 `DataLoader` 包装 dataset
- 配置 `collate_fn=NAGBatch.collate`

### Step 3: 修改推理循环
- 处理 NAGBatch 对象（取第一个元素）
- 保持原有的推理逻辑不变
- 保持原有的保存逻辑不变

### Step 4: 测试验证
- 运行小规模测试（`--limit 5`）
- 验证输出结果正确性
- 监控 GPU 利用率变化

## 代码修改详情

### 修改文件
- `batch_inference.py`

### 新增导入
```python
from torch.utils.data import DataLoader
from src.data import NAGBatch
```

### 新增参数
```python
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers')
parser.add_argument('--pin_memory', action='store_true', default=True,
                    help='Use pinned memory for faster GPU transfer')
parser.add_argument('--prefetch_factor', type=int, default=2,
                    help='Number of batches to prefetch per worker')
```

### 核心代码变更
```python
# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=NAGBatch.collate,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=args.num_workers > 0
)

# 修改推理循环
for batch in tqdm(dataloader, desc="Processing scenes"):
    # 从 NAGBatch 中提取单个 NAG
    if isinstance(batch, NAGBatch):
        batch = batch[0]
    # ... 后续逻辑保持不变
```

## 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| 多进程内存占用增加 | 低 | 可通过 `num_workers` 参数控制 |
| 首次迭代延迟 | 低 | `persistent_workers=True` 减少开销 |
| 兼容性问题 | 极低 | 项目已有 DataLoader 支持 |

## 验收标准

1. ✅ 推理结果与优化前一致
2. ✅ GPU 利用率提升到 60% 以上
3. ✅ 整体推理时间减少 30% 以上
4. ✅ 无内存溢出错误
