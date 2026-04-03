# 动态超点细化模块 (Dynamic Superpoint Refiner) Spec

## Why

当前 H-SPT 模块虽然实现了点级预测，但最终结果仍然回填到超点级别，无法真正实现"亚超点级"的精细分割。困难超点内部的边界点无法被单独分割，限制了分割精度的提升。

## What Changes

- 新增 **Superpoint Refiner (SPR)** 模块，基于点级预测结果动态重新划分困难超点
- 困难区域生成更细粒度的超点，简单区域保持原 Level 1 超点
- 构建混合粒度的超点图结构，支持后续聚类和分割
- 修改推理流程，输出真正的点级分割结果

## Impact

- Affected specs: H-SPT 三模块协作流程
- Affected code: 
  - `src/hspt/pipeline.py` - 新增 SPR 模块调用
  - `src/hspt/spr.py` - 新增 SPR 模块实现
  - `src/models/semantic.py` - 修改推理流程
  - `batch_inference.py` - 支持点级输出

## ADDED Requirements

### Requirement: 动态超点细化模块

系统 SHALL 提供动态超点细化功能，基于点级预测结果重新划分困难超点区域。

#### Scenario: 困难超点细化成功
- **GIVEN** AUS 筛选出 K 个困难超点
- **AND** RRH 输出点级预测 [K, N, C]
- **WHEN** 执行超点细化
- **THEN** 困难超点内部按语义边界被划分为多个细粒度超点
- **AND** 简单超点保持原 Level 1 结构
- **AND** 细化后的超点与简单超点形成混合层级

### Requirement: 混合粒度超点图构建

系统 SHALL 支持构建混合粒度的超点图结构。

### Requirement: 点级分割输出

系统 SHALL 支持输出点级分割结果。

## MODIFIED Requirements

### Requirement: H-SPT Pipeline 扩展

原有三模块流水线扩展为四模块：AUS → CAFM → RRH → SPR (新增)

### Requirement: 推理流程修改

推理时支持两种输出模式：
1. **超点级输出**：保持原有行为
2. **点级输出**：基于细化后的超点图，输出点级分割结果
