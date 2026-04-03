# Checklist

## Phase 1: 核心模块实现

- [ ] SPR 模块文件 `src/hspt/spr.py` 已创建
- [ ] `SuperpointRefiner` 类基础框架已实现
- [ ] `SPROutput` 数据结构已定义
- [ ] 语义边界检测函数已实现
- [ ] 边界点聚类算法已实现
- [ ] 超点分裂算法已实现
- [ ] 新超点 ID 分配逻辑已实现
- [ ] 邻接关系更新逻辑已实现

## Phase 2: Pipeline 集成

- [ ] `pipeline.py` 已添加 SPR 模块调用
- [ ] `run_hspt` 函数已支持 SPR 输出
- [ ] `HSPTOutput` 数据结构已更新
- [ ] `semantic.py` 已集成 SPR 输出
- [ ] 点级分割输出模式已实现
- [ ] 配置选项已添加

## Phase 3: 推理流程修改

- [ ] `batch_inference.py` 已支持点级输出
- [ ] `--point_level_output` 参数已添加
- [ ] 保存逻辑已修改，支持点级预测结果

## Phase 4: 测试与验证

- [ ] SPR 模块单元测试已通过
- [ ] H-SPT Pipeline 集成测试已通过
- [ ] 可视化验证细化效果符合预期
- [ ] 边界区域的分割精度有提升
- [ ] 简单区域的分割结果保持稳定

## 文档更新

- [ ] 代码注释已添加
- [ ] 函数文档字符串已完善
- [ ] 使用示例已更新
