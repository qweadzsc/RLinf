# stage2/resource 分支变更说明

## 1. 范围与统计

本文以 `upstream/main...HEAD` 为基线，覆盖当前 `stage2/resource` 分支中 **48 个已提交文件**的差异：约 **4,635 行新增、245 行删除**。内容按功能归类，并说明每个文件的职责。

本文不包含工作区尚未提交的本地试跑改动：SearchR1 FSDP 配置中的真实模型/数据路径、`max_steps: 2`，以及本地检索 server 脚本中的真实路径。

状态标记：

- **已启用**：当前代码路径或示例会使用。
- **实验/对齐**：用于硬件迁移、性能或数值对齐，不应作为默认训练配方理解。
- **遗留兼容**：保留了历史实现或兼容代码；需要后续清理或确认。

## 2. 功能一：Ascend 上的 FSDP2 模型加载与 attention 适配

这一组代码使 Hugging Face 模型能以 meta-device 初始化后由各 rank 分片加载，并为 Ascend 提供可调用的融合 attention 路径。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` | 已启用 | 新增 `init_model_with_meta_device` 配置校验与 FSDP2 专用加载路径；将 attention 实现改为由 `actor.model.attn_implementation` 控制，并在 `ascend_fusion` 时注册 Ascend attention patch。 |
| `rlinf/hybrid_engines/fsdp/strategy/fsdp2.py` | 已启用 | 新增 FSDP2 checkpoint loader：rank 0 在 CPU 读取 safetensors、按 DTensor 分片切分并散发到各 rank，避免每张卡完整加载 checkpoint；同时补齐 RoPE `inv_freq` 等不在 HF checkpoint 内的 meta buffer。 |
| `rlinf/hybrid_engines/fsdp/ascend_attention.py` | 已启用（Ascend） | 新增 Ascend 融合 attention 实现及 Qwen2 causal-mask patch；根据 packed sequence 的 `position_ids` 恢复真实序列长度，构造兼容融合算子的 mask。 |
| `rlinf/hybrid_engines/fsdp/strategy/fsdp.py` | 遗留兼容 | 加入 FSDP1 的 NPU meta 初始化辅助函数和一段注释掉的 FSDP 构造草案。当前主训练路径使用 FSDP2；这些内容不构成完整 FSDP1 Ascend 支持。 |
| `examples/reasoning/config/math/qwen2.5-1.5b-grpo-fsdp-ascend.yaml` | 实验/示例 | 新增 Qwen2.5-1.5B 数学 GRPO 的 Ascend FSDP 配置。 |
| `examples/reasoning/config/math/qwen2.5-32b-grpo-fsdp-ascend.yaml` | 实验/示例 | 新增 Qwen2.5-32B 数学 GRPO 的 Ascend FSDP 配置。 |
| `examples/reasoning/run_main_grpo_math_ascend.sh` | 示例 | 新增 reasoning 的 Ascend 启动入口，并保留可覆盖的 `MEGATRON_PATH`。 |

## 3. 功能二：训练—推理权重同步、FSDP actor 与动态 batch 正确性

这一组处理 FSDP/MAFSDP 训练中权重读取、reference policy、动态 rollout batch 及跨设备张量处理的问题。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `rlinf/hybrid_engines/sglang/common/sgl_scheduler.py` | 已启用 | 为 SGLang scheduler 的平台 API 增加兼容调用；调整 HF/FSDP 权重同步、设备缓存释放和权重校验逻辑，使 CUDA/NPU 可走同一套 scheduler 同步入口。 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | 已启用 | 新增 `_swap_to_ref_policy()`：通过 FSDP strategy 保存并恢复每 rank 的 shard state dict，在计算 reference logprob 时安全换权，而不是使用普通 PyTorch 的 state-dict swap。 |
| `rlinf/workers/actor/ma_fsdp_actor_worker.py` | 已启用 | 重构 MAFSDP 的训练步：支持 `BatchResizingIterator`、动态 micro-batch、空 batch、指标 all-reduce，以及 `prev_logprobs`/`recomputed_logprobs` 的兼容读取；reference logprob 复用 FSDP 专用换权逻辑。 |
| `rlinf/data/schema/reasoning_results.py` | 已启用 | 为 `DynamicRolloutResult` 增加/对齐动态 rollout 所需字段与访问方式，确保多智能体轨迹可被 MAFSDP 训练端读取。 |
| `rlinf/utils/data_iter_utils.py` | 已启用 | 动态 batch 切分时优先选择当前 accelerator，而不是从 CPU batch 推断设备；解决 NPU/CPU 混合状态下的设备错误。 |
| `rlinf/utils/distributed.py` | 已启用 | rollout 指标计算和 masked normalization 改为使用运行时 accelerator；避免为统计量意外创建 CPU/CUDA 张量，并处理空 batch 情形。 |
| `rlinf/utils/utils.py` | 已启用 | 修复 CPU state-dict 读取的 pin-memory 逻辑；新增 optimizer state 预热/初始化，且只重置本次初始化参数的 Adam step，避免误修改已有优化器状态。 |
| `tests/unit_tests/test_data_iter_utils.py` | 测试 | 覆盖运行时设备选择及 CPU batch fallback。 |

## 4. 功能三：SearchR1 Agent、工具调用与配置隔离

这一组将 SearchR1 的 Qwen 与 DeepSeek-R1 工具调用语义明确化，并新增 Ascend、FSDP parity 和资源监测的配置副本，尽量避免改变原 NVIDIA 配方语义。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `rlinf/agents/searchr1/searchr1_agent_loop.py` | 已启用 | 抽出 `encode_tool_resp()`：Qwen 使用纯文本 continuation，DeepSeek-R1 使用 native chat template；由 `use_native_tool_messages` 控制。 |
| `rlinf/agents/tool_call/parsers.py` | 已启用 | 新增/调整 SearchR1 Qwen 与 DeepSeek-R1 tool parser，识别 `<search>` 等模型协议中的工具调用并生成统一 `ToolRequest`；清除已废弃的 tool-response registry 装饰器，避免与上游 `encode_tool_resp()` 实现并存。 |
| `examples/agent/searchr1/config/train_deepseek_r1_ascend.yaml` | 示例 | 新增 DeepSeek-R1-Distill-Qwen 的 Ascend SearchR1 训练配置，启用 native tool message。 |
| `examples/agent/searchr1/config/train_qwen2.5_ascend.yaml` | 示例 | 新增 Qwen2.5 Ascend SearchR1 训练配置，包含 Ascend attention、精度、SGLang 等参数。 |
| `examples/agent/searchr1/config/train_qwen2.5_fsdp_parity.yaml` | 实验/对齐 | 新增 NVIDIA FSDP parity 配置，用于尽量匹配 Ascend 配置并定位数值/性能差异。 |
| `examples/agent/searchr1/config/train_qwen2.5_fsdp_resource.yaml` | 实验/资源监测 | 在 FSDP SearchR1 配方上启用资源监测与独立日志输出。 |
| `examples/agent/searchr1/config/train_qwen2.5.yaml` | 已启用 | 恢复/清理原 Qwen SearchR1 配置中的设备无关设置，并对齐新的 tool message 配置键。 |
| `examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml` | 已启用 | 保持 FSDP 基线配置，同时对齐工具消息和少量 rollout 配置。 |
| `examples/agent/searchr1/config/eval_qwen2.5.yaml` | 已启用 | 对齐评估配置中的 tool parser/tool message 语义及日志命名。 |
| `examples/agent/searchr1/run_train_ascend.sh` | 示例 | 新增 SearchR1 Ascend 启动脚本，使用可覆盖的模型并行路径。 |
| `tests/e2e_tests/agent/searchr1/qwen2.5-1.5b-fsdp.yaml` | 测试 | 修正官方 SearchR1 FSDP e2e 配置，使其匹配动态 rollout、FSDP2 和当前 tool message 语义。 |
| `tests/e2e_tests/agent/searchr1/qwen2.5-1.5b-megatron.yaml` | 测试 | 对齐 Megatron e2e 配置中的 tool parser 语义。 |
| `tests/e2e_tests/agent/searchr1/qwen2.5-1.5b-eval.yaml` | 测试 | 对齐评估 e2e 配置中的 tool parser 语义。 |

## 5. 功能四：本地 Qdrant 检索服务的 Ascend 适配

这一组将离线 embedding 建索引和在线 retriever server 分离为 Ascend 专用实现，避免让原 CUDA 路径承担 NPU 逻辑。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `examples/agent/tools/search_local_server_qdrant/build_index_ascend.py` | 示例 | 新增 NPU index builder：按 NPU device 初始化 encoder、分批编码并写入 Qdrant，同时等待 collection 进入 green 状态。 |
| `examples/agent/tools/search_local_server_qdrant/build_index_ascend.sh` | 示例 | 新增 Ascend 建索引启动脚本；提供 wiki、retriever、Qdrant 的环境变量占位符，并设置 `ASCEND_LAUNCH_BLOCKING`。 |
| `examples/agent/tools/search_local_server_qdrant/retriever_server_accel.py` | 已启用（Ascend） | 新增异步 retriever server：多 NPU encoder worker、Qdrant 检索、页面读取和 `/retrieve`/`/access` HTTP 接口。 |
| `examples/agent/tools/search_local_server_qdrant/launch_local_server_ascend.sh` | 示例 | 新增 Ascend retriever server 启动脚本。 |
| `examples/agent/tools/search_local_server_qdrant/build_index.py` | 已启用 | 原建索引实现改为设备自适应，提取 accelerator device 选择，减少 CUDA 硬编码。 |
| `examples/agent/tools/search_local_server_qdrant/build_index.sh` | 已启用 | 调整默认 retriever 名称/脚本参数以与当前 builder 一致。 |

## 6. 功能五：SGLang rollout 兼容、性能对齐与可复现性

这一组向 SGLang 透传必要的配置，使 CUDA graph、编译、采样后端与随机种子可显式控制，并修正 memory saver 与不同 SGLang 版本的参数兼容。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `rlinf/workers/rollout/sglang/sglang_worker.py` | 已启用 | 重构 `ServerArgs` 构造，按配置透传 `sampling_backend`、每 rank 随机种子、attention backend、torch compile、CUDA graph 和 memory saver；兼容不同 SGLang 版本的 EP 参数。 |
| `rlinf/agents/agentlightning/entrypoint.py` | 已启用 | 为 AgentLightning 启动补充 rollout server 初始化及后端配置传递。 |
| `rlinf/workers/agent/agentlightning_rollout_worker.py` | 已启用 | 对齐 AgentLightning rollout worker 的 engine HTTP 启动、请求参数和错误处理。 |

## 7. 功能六：资源监测与资源建模数据采集

这一组提供了 Agentic RL 一个 step 内的时间、CPU RSS、显存已分配/保留量与空闲显存数据，输出 JSONL，作为后续 placement、调度和迁移决策的输入。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `rlinf/utils/resource_monitor.py` | 已启用 | 新增资源采样、JSONL 写入和统计汇总工具；读取 `/proc/self/status` 的 RSS，并通过 accelerator 平台 API 获取显存指标。 |
| `rlinf/scheduler/worker/worker.py` | 已启用 | 新增统一的 `get_resource_snapshot()` Worker API，让 runner 能向各 worker 请求资源状态。 |
| `rlinf/runners/agent_runner.py` | 已启用 | 在 AgentRunner 生命周期采集 actor、rollout、agent/reward 的资源快照，记录阶段耗时并写入资源日志。 |
| `tests/unit_tests/test_resource_monitor.py` | 测试 | 使用 fake accelerator 覆盖资源快照、汇总与 JSONL 写入。 |

## 8. 功能七：AgentLightning Calc-X 的 FSDP/Ascend 示例

这一组为 Calc-X 多轮工具环境提供可运行的 FSDP 配方与 Ascend 差异配置，用于补充 SearchR1 之外的 Agent 场景。

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `examples/agent/agentlightning/calc_x/config/base_calc_x.yaml` | 已启用 | 调整 Calc-X 基础配置的运行/日志默认值。 |
| `examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn-fsdp.yaml` | 示例 | 新增 Qwen2.5-1.5B、EngineHTTP、多轮、FSDP 的 Calc-X 主配置。 |
| `examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn-fsdp_ascend.yaml` | 示例 | 新增上述配方的 Ascend 变体，覆盖精度、attention 和 backend 差异。 |
| `examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn-fsdp_a800_test.yaml` | 测试 | 针对单卡 A800 的 FSDP 测试覆盖配置。 |
| `examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn_a800_test.yaml` | 测试 | 针对单卡 A800 的非 FSDP/基础 engine 测试覆盖配置。 |

## 9. 功能八：安装、忽略规则与工程卫生

| 文件 | 状态 | 变更目的 |
|---|---|---|
| `requirements/install_ascend.sh` | 实验/安装 | 新增 Ascend 独立安装脚本：管理 uv 环境、torch/torch_npu wheel、SGLang 与 sgl-kernel-npu 源码安装和缓存目录。该脚本仍应在 CANN/torch_npu 版本稳定后考虑并入统一安装流程。 |
| `.gitignore` | 已启用 | 忽略本地生成的测试、缓存或辅助产物，减少误提交。 |

## 10. 关联关系与建议

1. **核心训练链路**：`FSDPModelManager`/`FSDP2Strategy` 负责模型创建与加载，`FSDPActor`/`MAFSDPActor` 负责训练和 reference policy，`Scheduler`/`SGLangWorker` 负责权重同步与 rollout。
2. **Agent 链路**：SearchR1 的 `Searchr1AgentLoopWorker` 解析工具调用、编码工具返回；Qdrant server 为 SearchR1 提供本地知识检索。
3. **资源链路**：`AgentRunner` 调用 `Worker.get_resource_snapshot()`，由 `resource_monitor.py` 写出 JSONL；资源配置文件只是在相同训练配方上开启该采集。
4. **后续清理优先级**：应优先核对 `fsdp.py` 中 FSDP1 的历史 NPU 草案、独立 `install_ascend.sh` 的安装策略，以及 parity/A800 测试配置是否需要长期保留。它们不应与稳定的默认 CUDA 路径混淆。
