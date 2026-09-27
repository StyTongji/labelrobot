# LabelRobot

[English](README.md) | **简体中文**

LabelRobot 是一个面向 **LeRobot 格式机器人（机械臂）操作数据集** 的本地轨迹查看与 **细粒度数据标注** 工具。

它只读地打开你的 LeRobot 数据集，把标注结果保存在源数据集**之外**的旁挂（sidecar）目录中，导出**不可变的 Parquet 快照**作为训练输入，并通过 PyTorch Dataset 包装器把标签喂给下游策略训练。

> **源数据集永不被写入。** 所有新增、修改、撤销都只发生在 sidecar 目录里。

## 它解决什么问题

LeRobot 数据集本身只提供原始信号：逐帧 `observation.state` / `action`、多路相机视频，以及一个**粗粒度**的 task 字符串。但要做过程奖励建模、失败归因、分层策略或数据质量筛选，你需要的是帧级与段级的结构化标签——"这一段的子任务是 `grasp`"、"第 42 帧发生了 `object_slip`"、"这次 `align_gripper` 的预期结果是 `object_secured`，但验证结论是 `not_met`"。

LabelRobot 提供的就是这一层：一个本地跑、可交互、带一致性校验的标注界面，加上一条从标注到训练样本的完整通路。

## 核心能力

### 1. 只读兼容 LeRobot 数据集

- 支持 **LeRobot v2.0 / v2.1 / v3.0**，通过 `meta/info.json` 中的 `codebase_version` 自动分派 reader。
- 处理 v2/v3 两代目录约定差异：`data_path` / `video_path` 模板、chunk / file 索引、`meta/episodes.jsonl` 与 `meta/episodes/` 两种索引形态。
- 正确处理**共享 Parquet/MP4**（一个文件内含多条 episode）与 **camera 时间偏移**（`from_timestamp` / `to_timestamp`），支持 episode 子集。
- 默认开启离线模式（`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`），不联网、不上传任何数据。
- 数据集指纹：对 `meta/` 内容与 `data/`、`videos/` 文件尺寸/时间戳做 SHA-256，标注结果可追溯到确切的数据集版本。

### 2. 细粒度标注体系

| 标注类型 | scope | 含义 |
| --- | --- | --- |
| `subtask` | `segment` | 层级子任务段，靠 `parent_id` 表达任务分解树 |
| `recovery` | `segment` | 失败恢复段（重试、重规划、人工介入等） |
| `observed_event` | `point` / `segment` | **实际发生**的事件，如抓取失败、物体滑落、碰撞 |
| `expected_outcome` | `point` | **人工回溯**的预期结果，附带验证窗口 |
| `verification` | `point` | 将预期与观察关联起来，结论为 `met` / `not_met` / `uncertain` |
| `progress` | `frame_value` | 任务进度锚点，值域 `[0, 1]` |
| `episode_evaluation` | `episode` | 整条 episode 的评分（`0`–`5`）与成功标记 |

分段统一使用 **左闭右开** 区间 `[start_frame, end_frame_exclusive)`。

### 3. 一致性校验

标注在写入时即被校验，而不是留给下游发现：

- 子段必须**包含在父段之内**；同级段默认**不允许重叠**。
- 每条 episode 只能有一个 `episode_evaluation`；同一帧只能有一个 `progress` 锚点。
- `progress` 值必须在 `[0, 1]`，评分必须在 `[0, 5]`。
- `verification` 关联的预期与观察**必须属于同一 episode**，且验证不得早于其证据帧。
- `expected_outcome` 的验证窗口必须是合法帧区间。
- 所有帧区间对照数据集真实长度校验，`camera_key` 必须存在于数据集 features 中。

### 4. 并发安全与可回溯

- **SQLite 是编辑的唯一权威**，所有编辑以命令（command）形式提交，携带 `expected_revision`；版本不匹配时抛出 `RevisionConflict`，避免多人/多标签页互相覆盖。
- 界面提供会话级 **undo / redo**。
- **快照是训练输入**：`snapshot` 命令把当前状态原子地导出为 `snapshots/<id>/{annotations.parquet, links.parquet, info.json}`，导出完成后不可变，训练侧永远读到一致的数据。

### 5. 训练闭环

`LabelRobotDatasetWrapper` 包装任意包含 `episode_index` 与 `frame_index` 的基座 Dataset，为每个样本附加标签，并提供显式的未来信息查询接口。

## 安装

需要 Python ≥ 3.11。

```bash
git clone git@github.com:StyTontji/labelrobot.git
cd labelrobot
pip install -e .
```

前端产物已随仓库提供（`src/labelrobot/static/`），仅做 Python 侧使用无需安装 Node。

## 快速开始

```bash
# 启动本地界面（默认自动打开浏览器）
python -m labelrobot /path/to/dataset
```

标注默认写到 `/path/to/dataset.labelrobot`，源数据集只读打开。

```bash
# 不自动打开浏览器
python -m labelrobot /path/to/dataset --no-browser

# 自定义标注目录 / 监听地址
python -m labelrobot /path/to/dataset --annotations /path/to/annotations --host 127.0.0.1 --port 8765

# 导出训练用快照
python -m labelrobot snapshot /path/to/dataset
```

`snapshot` 会打印生成的快照 ID。

## 界面功能

浏览器界面（默认 `http://127.0.0.1:8765`）支持：

- episode 列表与浏览，多相机视频与逐帧图像查看
- **精确到帧的步进**，同步显示 timestamp、state、action 数值
- 层级 subtask 的创建、修改、删除
- 失败恢复段、观察事件、预期结果与验证链接的标注
- 进度锚点与 episode 评分 / 成功标记
- 会话内 undo / redo

## HTTP API

后端为 FastAPI，同时服务前端静态资源。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/dataset` | 数据集元信息 |
| `GET` | `/api/episodes` | episode 列表 |
| `GET` | `/api/episodes/{episode_index}/index` | 该 episode 的帧索引与时间戳 |
| `GET` | `/api/episodes/{episode_index}/rows` | 按帧区间读取 state/action 等列 |
| `GET` | `/api/episodes/{episode_index}/media` | 该 episode 的相机媒体片段 |
| `GET` | `/api/media/{resource_id}` | 媒体流 |
| `GET` | `/api/episodes/{episode_index}/frames/{frame_index}/{camera_key}` | 单帧图像 |
| `GET` | `/api/tools` | 当前项目的标注词表 |
| `GET` | `/api/episodes/{episode_index}/annotations` | 该 episode 的全部标注与链接 |
| `POST` | `/api/annotation-commands` | 提交编辑命令（带 `expected_revision`） |
| `POST` | `/api/snapshots` | 创建快照 |

## 训练接入

```python
from labelrobot.training import LabelRobotDatasetWrapper

wrapped = LabelRobotDatasetWrapper(base, "/path/to/dataset.labelrobot")

sample = wrapped[0]
# 附加到基座样本上的字段：
#   annotation.subtask            int64   子任务词表索引，未知为 -1
#   annotation.subtask_valid      bool
#   annotation.observed_event     bool[]  多热事件向量
#   annotation.observed_event_valid bool
#   annotation.progress           float32 锚点间线性插值的进度
#   annotation.progress_valid     bool
#   annotation.episode_score      int64   未知为 -1
#   annotation.episode_score_valid bool
#   annotation.episode_success    bool
#   annotation.episode_success_valid bool

path = wrapped.get_subtask_path(episode_index=0, frame_index=42)
```

可用的查询接口：

```python
wrapped.get_episode_annotations(episode_index)
wrapped.get_frame_annotations(episode_index, frame_index)
wrapped.get_segments(episode_index)
wrapped.get_events(episode_index)
wrapped.get_subtask_path(episode_index, frame_index)
wrapped.get_transition_targets(episode_index, start_frame, end_frame_exclusive)
```

> **避免时间信息泄漏的设计：** 默认样本只包含当前帧可见的标签。未来的 `observed_event` 与 `verification` 结论**不会**注入当前帧样本，只能通过 `get_transition_targets` 显式获取。同时 `progress` 只在锚点之间线性插值，绝不外推。

## 标注词表

默认词表位于 `src/labelrobot/assets/default_tools.json`，面向机械臂操作任务：

```json
{
  "subtask": ["approach_object", "align_gripper", "grasp", "lift", "transport", "align_target", "release", "verify"],
  "recovery": ["retry_grasp", "reapproach", "realign", "replan", "human_intervention"],
  "observed_event": ["grasp_miss", "object_slip", "collision", "object_drop", "wrong_target", "wrong_place", "other"],
  "expected_outcome": ["object_secured", "object_at_target", "contact_established", "motion_completed"]
}
```

训练侧的 `annotation.subtask` 按 `subtask` + `recovery` 两个列表拼接后的顺序编码，`annotation.observed_event` 按 `observed_event` 列表顺序编码；命中不了词表时索引为 `-1`，并由对应的 `*_valid` 布尔位标记。

## 目录结构

```
labelrobot/
├── src/labelrobot/
│   ├── __main__.py          # CLI 入口（serve / snapshot）
│   ├── api.py               # FastAPI 应用与端点
│   ├── storage.py           # SQLite 标注存储、校验、快照导出
│   ├── resolver.py          # 只读快照解析（供训练使用）
│   ├── training.py          # PyTorch Dataset 包装器
│   ├── models.py            # 数据模型（Annotation / Link / DatasetInfo ...）
│   ├── readers/             # LeRobot v2 / v3 只读读取器
│   ├── assets/              # 默认标注词表
│   └── static/              # 前端构建产物
├── frontend/                # React + Vite + TypeScript 界面源码
├── examples/                # 训练接入示例
└── tests/                   # pytest 测试
```

## 开发

前端源码在 `frontend/`，构建产物输出到 `src/labelrobot/static/`：

```bash
cd frontend
npm install
npm run dev      # 开发服务器，/api 代理到 127.0.0.1:8765
npm run build    # 重新生成 src/labelrobot/static/
```

改前端后需要重新 `npm run build`，Python 侧的 `python -m labelrobot` 才会提供新的界面。

## 测试

```bash
pip install -e ".[test]"
pytest
```

## 设计取舍

- **只读源数据 + 旁挂标注**：标注不污染原始数据集，可以随时丢弃重来，也便于把标注目录单独备份或共享。
- **SQLite 编辑 + Parquet 快照分离**：编辑需要事务与并发控制，训练需要不可变、可并行读取的列式文件，两者用快照解耦。
- **默认不注入未来信息**：标签体系里天然包含"后来发生了什么"，因此默认样本严格只看当前帧，把因果信息留给显式的 transition targets。
- **本地优先**：默认绑定 `127.0.0.1` 并强制离线，数据不出机器。

## 路线图

已完成 R0–R3（只读读取与轨迹检查 → 层级 subtask → 事件推理标注 → 训练闭环）。以下方向在真实科研使用证明需要后再加入：GUI 词表编辑器、schema 迁移、PCHIP 插值、自动转码、dataset 物化、模型预测导入、实时采集标记、大规模分布式优化。
