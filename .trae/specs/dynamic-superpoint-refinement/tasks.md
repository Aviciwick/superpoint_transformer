# Tasks

## Phase 1: 核心模块实现

- [ ] Task 1: 创建 SPR 模块基础结构
  - [ ] SubTask 1.1: 创建 `src/hspt/spr.py` 文件
  - [ ] SubTask 1.2: 实现 `SuperpointRefiner` 类基础框架
  - [ ] SubTask 1.3: 定义输入输出数据结构 `SPROutput`

- [ ] Task 2: 实现语义边界检测
  - [ ] SubTask 2.1: 实现基于点级预测的语义边界检测函数
  - [ ] SubTask 2.2: 实现边界点聚类算法
  - [ ] SubTask 2.3: 添加边界平滑后处理

- [ ] Task 3: 实现超点分裂算法
  - [ ] SubTask 3.1: 实现基于语义标签的超点分裂
  - [ ] SubTask 3.2: 实现新超点 ID 分配逻辑
  - [ ] SubTask 3.3: 处理边界情况（单标签超点、空超点等）

- [ ] Task 4: 实现邻接关系更新
  - [ ] SubTask 4.1: 实现细粒度超点之间的邻接关系构建
  - [ ] SubTask 4.2: 实现细粒度与粗粒度超点的邻接关系构建
  - [ ] SubTask 4.3: 更新 edge_index 和 edge_attr

## Phase 2: Pipeline 集成

- [ ] Task 5: 修改 H-SPT Pipeline
  - [ ] SubTask 5.1: 在 `pipeline.py` 中添加 SPR 模块调用
  - [ ] SubTask 5.2: 修改 `run_hspt` 函数，支持 SPR 输出
  - [ ] SubTask 5.3: 更新 `HSPTOutput` 数据结构

- [ ] Task 6: 修改 Semantic 模型
  - [ ] SubTask 6.1: 在 `semantic.py` 中集成 SPR 输出
  - [ ] SubTask 6.2: 实现点级分割输出模式
  - [ ] SubTask 6.3: 添加配置选项控制输出模式

## Phase 3: 推理流程修改

- [ ] Task 7: 修改推理脚本
  - [ ] SubTask 7.1: 在 `batch_inference.py` 中支持点级输出
  - [ ] SubTask 7.2: 添加 `--point_level_output` 参数
  - [ ] SubTask 7.3: 修改保存逻辑，支持点级预测结果

- [ ] Task 8: 测试与验证
  - [ ] SubTask 8.1: 单元测试 SPR 模块
  - [ ] SubTask 8.2: 集成测试 H-SPT Pipeline
  - [ ] SubTask 8.3: 可视化验证细化效果

# Task Dependencies

- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 5]
- [Task 7] depends on [Task 6]
- [Task 8] depends on [Task 7]

# Parallelizable Work

- Task 1, Task 2, Task 3 可并行开始（基础框架可独立开发）
- Task 8 的单元测试可与 Task 5-7 并行进行
