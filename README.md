# 大模型技术全景解析与Qwen/DeepSeek实践

## 📚 专题目录
### 1. 架构设计
- **Decoder-only架构**
  - Dense结构 vs MOE结构
  - Normalization技术（RMSNorm, LayerNorm）
  - Tokenizer设计与BPE算法
  - 位置编码（RoPE, ALiBi）
  - Attention机制演进
    - MHA (Multi-Head Attention)
    - GQA (Grouped-Query Attention) 
    - MLA (Multi-Query Latent Attention)

### 2. 训练方法论
- **预训练阶段**
  - 数据Pipeline构建
  - 损失函数设计
- **继续训练**
  - 领域适应训练
- **微调技术**
  - 全参数微调
  - LoRA/P-Tuningv2
- **RLHF框架**
  - PPO算法实现
  - DPO直接偏好优化
  - GRPO约束优化

### 3. 并行训练策略
| 并行类型 | 通信特点 | 适用场景 |
|---------|---------|---------|
| DP (Data Parallel) | AllReduce梯度同步 | 单机多卡 |
| TP (Tensor Parallel) | 张量分片计算 | 大矩阵运算 |
| PP (Pipeline Parallel) | 流水线气泡优化 | 超长模型 |
| 3D Parallelism | 混合并行策略 | 超大规模训练 |

### 4. Qwen专题
- 模型架构特点
- 训练数据构建
- 量化部署方案
- 微调最佳实践

### 5. DeepSeek专题
- MoE实现解析
- 动态负载均衡
- 专家并行策略
- 推理性能优化
