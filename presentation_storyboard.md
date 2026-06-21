# D2A2 + MLOps 汇报 Storyboard

## Global Style

- Style: Academic Engineering, Course Project, Dark Slate, Blue-Teal Accent
- Background: dark slate / charcoal, clean grid only where流程图需要对齐
- Accent: muted blue 表示主流程，teal 表示最优模型或关键结论，amber 表示风险和 TBD 占位
- Typography: 课程项目汇报风格，标题清楚，正文克制，避免发布会式视觉效果
- Visual Tone: practical, technical, evidence-oriented
- Rule: 不编造实验结果；标记 TBD 的截图和图表必须由真实实验素材替换
- Target Duration: 5 minutes total
- Target Page Count: 9 pages

---

## Page 1

### Title

D2A2 + MLOps：面向深度超分辨率的机器学习系统

### Goal

让观众知道项目主题、任务方向和汇报范围。

### Narrative

第一页只作为标题页，不展开工具细节，也不放 bullet 列表。它需要建立一个清晰印象：本项目以 D2A2 为核心模型，围绕深度超分辨率训练过程做 MLOps 化改造，并在后面展示实验管理和生成性能结果。

### Slide Type

Cover

### Layout

简洁封面。页面中央偏上放主标题，标题下方放副标题“深度超分辨率模型训练与实验管理”。左下角放课程项目、组名或汇报人信息。右侧可以放一条很淡的抽象系统线条作为背景，不展示工具链 bullet。

### Visual Focus

项目主题标题

### Visual Description

使用深灰背景和少量蓝绿色细线，不使用强发光效果。视觉中心是中文主标题，副标题用较小字号。右侧背景线条只作为暗纹，不能喧宾夺主。

### Assets

- 无，需要 PPT 直接排版

### Slide Content

- 无 bullet 列表
- 主标题：D2A2 + MLOps
- 副标题：面向深度超分辨率的机器学习系统
- 页脚：课程项目 / 汇报人 / 日期

### Speaker Notes

本项目主题是 D2A2 + MLOps。D2A2 负责核心深度超分辨率任务，MLOps 负责把科研训练流程变成可追踪、可复现、可比较的实验系统。后面会按任务背景、模型、MLOps 架构、训练流程、实验追踪和生成性能结果展开。

### Estimated Time

15 seconds

---

## Page 2

### Title

任务背景：为什么需要深度超分辨率

### Goal

用轻量文字说明问题，不依赖图片素材。

### Narrative

这一页只用文字建立背景：深度图能提供场景几何信息，但低成本深度传感器得到的 depth 往往分辨率低、噪声多、边界不清晰。RGB 图像更容易获取，并且包含边缘和结构信息。因此任务目标是用高分辨率 RGB 引导低分辨率 depth，恢复更高质量的 depth。

### Slide Type

Problem Statement

### Layout

纯文字结构。页面左侧放一个大问题句：“低质量 depth 难以满足视觉任务”。右侧用三条短句列出原因。底部用一行公式展示任务定义：LR depth + HR RGB -> HR depth。

### Visual Focus

任务定义公式

### Visual Description

不放真实图片。使用三段文字卡片：Low-resolution、Noisy、Blurred boundary。底部公式用等宽字体和 muted blue 箭头。整体应轻量、清楚，不需要额外 assets。

### Assets

- 无，需要 PPT 绘制文字卡片和公式

### Slide Content

- 低成本 depth：低分辨率、噪声、边界模糊
- RGB 更容易获取
- RGB 包含边缘与结构信息
- 任务：LR depth + HR RGB -> HR depth

### Speaker Notes

我们的任务来自高质量深度重建需求。低成本深度图通常分辨率低、边界模糊，并且可能有噪声。RGB 图像更容易获得，也包含丰富结构信息，所以可以作为 guidance，引导低分辨率 depth 恢复成更高质量的 depth。

### Estimated Time

25 seconds

---

## Page 3

### Title

D2A2：核心重建模型

### Goal

解释 D2A2 在系统中承担核心建模角色，并说明三路输入。

### Narrative

这一页把任务连接到模型。D2A2 使用 HR RGB、LR depth 和单目深度估计得到的 MDE pseudo-depth 三路输入，输出高分辨率 depth。这里不展开完整论文结构，只保留和系统相关的接口信息：模型训练和推理都依赖三路输入，这会影响后面的实验记录和模型推理封装。

### Slide Type

Architecture Diagram

### Layout

中心结构图。左侧三张输入卡片竖排：HR RGB、LR depth、MDE pseudo-depth；中间是 D2A2 模块；右侧是 HR depth 输出。底部放模型调用形式：model(guidance, target, mde)。

### Visual Focus

D2A2 三输入模型结构

### Visual Description

三路输入可以用简化卡片表示，不强制使用真实图片。三条蓝色箭头汇入 D2A2 模块，输出箭头指向 HR depth。模块边框使用 muted blue，不做强烈发光。底部接口说明用小号等宽字体。

### Assets

- Optional TBD: `figures/d2a2_input_output.png`：D2A2 输入输出示意图，用来展示 HR RGB、LR depth、MDE pseudo-depth 三路输入进入 D2A2，并输出 HR depth；如果没有现成图，可以直接用 PPT 形状绘制。

### Slide Content

- HR RGB guidance
- LR depth target
- MDE pseudo-depth prior
- D2A2 输出 HR depth
- 三输入接口影响实验记录与推理封装

### Speaker Notes

D2A2 是我们的核心模型方案。它不是单输入模型，而是同时使用 RGB、低分辨率 depth 和 MDE pseudo-depth。这个接口信息很重要，因为后面无论是训练记录、最优模型选择，还是推理封装，都要围绕这三路输入来组织。

### Estimated Time

30 seconds

---

## Page 4

### Title

为什么需要 MLOps

### Goal

用简化文字说明普通训练流程的问题，不做复杂对比图。

### Narrative

单次训练可以手工跑通，但课程项目需要比较多组实验。传统流程里，参数在命令行，数据版本只是一条路径，checkpoint 在本地目录，指标在日志中。实验一多，就很难复盘某个模型来自哪份代码、哪份数据、哪组参数，以及为什么它是最优。因此需要把训练流程工程化。

### Slide Type

Problem Statement

### Layout

一列文字 + 一个简单图例。页面上方放一句核心问题：“实验多了以后，结果缺少证据链”。下方用四个小图例表示 Parameters、Data、Checkpoint、Metrics 分散。右下角放结论箭头：“需要统一实验记录”。

### Visual Focus

实验信息分散的四个图例

### Visual Description

不使用图片素材。四个小图例用简单线框卡片即可，每张卡片只放一个词和一个短说明。卡片之间不需要复杂连线，只需要表现“分散”。右下角用 muted blue 箭头指向“统一到 run”。

### Assets

- 无，需要 PPT 绘制文字卡片和简单图例

### Slide Content

- 参数、数据、checkpoint、指标容易分散
- 数据路径不等于数据版本
- 测试结果可能和 checkpoint 脱节
- 模型选择需要完整证据链

### Speaker Notes

项目重点不只是把模型跑起来。实验变多后，参数、数据版本、checkpoint 和指标会分散，最终很难解释为什么选中某个模型。MLOps 的作用就是把这些信息组织进同一个可查询、可比较的实验记录。

### Estimated Time

25 seconds

---

## Page 5

### Title

系统架构：DVC + MLflow + Optuna

### Goal

给出 MLOps 主体架构，并说明每个工具在训练实验链路中的职责。

### Narrative

这一页是全局架构。Git 和 DVC 固定代码与数据版本；训练脚本执行 D2A2 的 train、validate、test；MLflow 记录参数、指标、checkpoint、日志和 DVC metadata；Optuna 组织多组 trial 搜索超参数。结论是这些工具不是并列展示，而是共同构成实验和模型选择证据链。

### Slide Type

Architecture Diagram

### Layout

横向泳道图，分为 Versioning、Training、Tracking、HPO、Evaluation 五段。每段放一个主模块和 1 到 2 个关键输出。底部用细线标出 Evidence Chain。

### Visual Focus

D2A2 + MLOps 总体架构

### Visual Description

主图从左到右流动：Git commit 和 DVC dataset hash 进入训练脚本；训练脚本调用 D2A2 并输出 checkpoints；MLflow run 接收 params、metrics、artifacts；Optuna parent run 管理 nested trials；最终进入 generation quality 和 inference performance 展示。颜色保持深灰底、蓝色数据流、teal 高亮 best model。

### Assets

- 无，需要 PPT 绘制架构图

### Slide Content

- DVC 绑定数据快照
- D2A2 执行 train / validate / test
- MLflow 记录实验证据
- Optuna 搜索超参数
- 实验结果展示生成质量与性能

### Speaker Notes

整体架构可以看成一条证据链：代码和数据先被固定，训练验证测试进入统一流程，MLflow 记录每次实验，Optuna 组织多组参数搜索，最后通过实验结果展示模型生成质量和推理性能。系统要回答的是：模型从哪里来、为什么选它、结果如何复现和比较。

### Estimated Time

35 seconds

---

## Page 6

### Title

训练流程与模型选择

### Goal

说明训练、验证、测试如何进入同一个闭环，并如何选择 best model。

### Narrative

训练流程被改造成连续 pipeline：build model、build dataloader、train each epoch、validate by interval、保存 val_mean_rmse 最优的 best_parameter、训练结束后加载 best_parameter 做 test，并把 best_model_test_rmse_for_gate 写回 MLflow。最终模型不是最后一轮 checkpoint，而是验证集最优权重加跨 trial RMSE 选择。

### Slide Type

Pipeline Diagram

### Layout

上半部分是 train -> validate -> best_parameter -> test -> log metric 的流程图。下半部分放两层选择逻辑：trial 内按 val_mean_rmse 选权重，trial 间按 objective_rmse 选 best trial。

### Visual Focus

best_parameter 选择闭环

### Visual Description

流程节点用矩形，validate 后接判断菱形：“lower val_mean_rmse?”，是则保存 best_parameter。best_parameter 用 teal 高亮，并连接到 test 阶段。右侧 MLflow run 卡片接收最终 gate metric。底部两层选择逻辑用简洁漏斗图表现。

### Assets

- 无，需要 PPT 绘制流程图

### Slide Content

- 训练中定期验证
- 按 val_mean_rmse 保存 best_parameter
- 加载最优 checkpoint 再测试
- 记录 best_model_test_rmse_for_gate
- 跨 trial 按 RMSE 选择模型

### Speaker Notes

训练和测试不再是两个孤立入口。训练过程中定期验证，如果 val_mean_rmse 更低，就保存 best_parameter。训练结束后加载这个 checkpoint 做测试，并把测试 RMSE 写回同一个 MLflow run。跨 trial 时再比较 objective_rmse，选择最终模型。

### Estimated Time

35 seconds

---

## Page 7

### Title

实验追踪：DVC、MLflow 与 Optuna

### Goal

用一页讲清楚三个关键 MLOps 工具的技术细节。

### Narrative

这一页集中展示三类实验记录。DVC 记录 NYUv2 数据快照，包括 hash、size、nfiles；MLflow 记录 params、metrics、artifacts 和 tags；Optuna 以 minimize RMSE 为目标，运行 10 个 trial，每个 trial 是 MLflow nested run。结论是数据、实验和调参被绑定到统一追踪体系中。

### Slide Type

Workflow

### Layout

三栏结构。左栏 DVC，展示 .dvc pointer；中栏 MLflow，展示 run 详情截图占位；右栏 Optuna，展示 trial 搜索过程和 history 图占位。每栏底部写一句该工具回答的问题。

### Visual Focus

MLflow 实验记录面板

### Visual Description

中间 MLflow 截图占最大宽度，是本页视觉中心；左侧 DVC 用小代码卡片展示 md5 / size / nfiles；右侧 Optuna 用 trial 列表和 best trial 高亮。所有真实 UI 或曲线必须由后续素材替换，不能生成假截图。缺素材时保留明确 TBD 占位框。

### Assets

- `datasets/NYUv2.dvc`：真实 DVC 数据指针文件，用来展示数据 hash、size、nfiles、path 等数据版本信息。
- TBD: `figures/mlflow_run_detail.png`：真实 MLflow run 详情页截图，用来展示 params、metrics、artifacts、tags 等实验记录。
- TBD: `figures/optuna_history.png`：真实 Optuna 调参过程图，用来展示 trial 的 objective RMSE 变化或 best value 曲线。
- TBD: `figures/mlflow_nested_runs.png`：真实 MLflow nested runs 截图，用来展示 Optuna parent run 与多个 trial run 的对应关系。

### Slide Content

- DVC：dataset hash、size、nfiles
- MLflow：params、metrics、artifacts、tags
- Optuna：10 trials，minimize RMSE
- nested runs 让每个 trial 可比较
- 必须替换为真实截图

### Speaker Notes

DVC 让实验绑定到明确数据快照，而不是只有 dataset_dir。MLflow 是实验主索引，记录参数、训练和测试指标、checkpoint、日志和 DVC metadata。Optuna 负责调参，当前目标是 minimize RMSE，每个 trial 都作为 MLflow nested run，方便后续比较。

### Estimated Time

45 seconds

---

## Page 8

### Title

实验展示：生成质量与推理性能

### Goal

展示真实生成性能结果，并明确指标和图表位置。

### Narrative

这一页用于实验展示，重点不再讲工具，而是展示模型生成质量和核心实验指标。建议从两类证据组织：生成质量样例、性能指标。质量样例展示 RGB / LR depth / generated HR depth / reference 或可视化对比；性能指标展示 RMSE 等真实数值。所有结果必须来自真实实验，不由 PPT 工具生成。

### Slide Type

Results

### Layout

上半部分放大图对比条：RGB、LR depth、Generated HR depth、Reference/GT。下半部分放指标表或简洁柱状图。右上角小标签标明 dataset 和 model checkpoint。

### Visual Focus

真实生成结果对比图

### Visual Description

主视觉必须是真实生成结果图，不使用 AI 生成替代图。四列图像保持同一场景、同一尺寸，Generated HR depth 用 teal 边框强调。下方指标区域放真实 RMSE 或其他质量指标，若暂时没有结果则保留 TBD 占位。不要写模拟数值，也不要自动画假图表。

### Assets

- TBD: `figures/generation_quality_comparison.png`：真实生成质量对比图，建议同一行展示 RGB、LR depth、Generated HR depth、Reference/GT。
- TBD: `figures/generation_metrics_table.png`：真实实验指标表或截图，用来展示 RMSE 或其他生成质量指标。
- Optional TBD: `figures/depth_error_map.png`：可选误差图，用来展示生成 depth 与 Reference/GT 的误差分布。

### Slide Content

- 展示真实生成 depth 样例
- 对比 RGB / LR depth / output / reference
- 报告 RMSE 或质量指标
- 不编造性能数值

### Speaker Notes

实验部分主要展示生成质量。这里建议放真实样例图，最好是同一场景下的 RGB、低分辨率 depth、模型生成结果和参考结果。下方再放真实质量指标，例如 RMSE。如果还没有最终结果，这页只保留占位，后续用实际实验图替换，不能让生成工具自动编数值。

### Estimated Time

65 seconds

---

## Page 9

### Title

总结：从模型训练到可管理实验

### Goal

总结项目贡献，并收束到课程项目价值。

### Narrative

最后一页强调项目产出：D2A2 提供深度超分辨率模型能力；DVC 绑定数据版本；MLflow 统一参数、指标、日志和模型产物；Optuna 支持可比较的调参；实验页展示生成质量和推理性能。结论是项目从“能训练模型”推进到“能管理、复现、选择和评估模型”。

### Slide Type

Conclusion

### Layout

中心放一句总结，周围四个小模块环绕：Model、Data、Tracking、Tuning、Evaluation。底部放 future work：完整 DVC pipeline、持久化 Optuna storage、更多生成性能实验。

### Visual Focus

课程项目能力矩阵

### Visual Description

使用简洁能力矩阵，不用复杂闭环动画。每个模块用同尺寸深色卡片，蓝色细边框，Evaluation 使用 teal 高亮表示实验结果承接全链路。底部 future work 用小号文字，不影响主结论。

### Assets

- 无，需要 PPT 绘制能力矩阵

### Slide Content

- D2A2：核心重建模型
- DVC + MLflow：可复现实验证据
- Optuna：可比较的超参数搜索
- 实验：生成质量与推理性能

### Speaker Notes

总结来说，我们以 D2A2 为核心模型，把传统训练流程扩展成 MLOps 实验系统。DVC 管数据版本，MLflow 管实验记录，Optuna 管调参，实验部分展示生成质量和性能。项目价值在于让模型训练结果可追踪、可复现、可比较，并能被真实实验结果验证。

### Estimated Time

25 seconds

---

## Assets 说明

- `figures/d2a2_input_output.png`：可选的 D2A2 输入输出示意图。内容应展示 HR RGB、LR depth、MDE pseudo-depth 三路输入进入 D2A2，并输出 HR depth。如果没有现成图，可以由 PPT 直接绘制，不需要生成图片。
- `datasets/NYUv2.dvc`：真实 DVC 数据指针文件，用于在 Page 7 展示数据 hash、size、nfiles、path 等元信息。
- `figures/mlflow_run_detail.png`：真实 MLflow run 详情页截图。最好能看到 params、metrics、artifacts 或 tags 中的关键字段。
- `figures/optuna_history.png`：真实 Optuna 调参过程图，例如 optimization history、trial value 曲线或 best value 曲线。
- `figures/mlflow_nested_runs.png`：真实 MLflow nested runs 截图，用于展示 parent run 与多个 trial run 的关系。
- `figures/generation_quality_comparison.png`：真实生成质量对比图。建议同一行展示 RGB、LR depth、Generated HR depth、Reference/GT。
- `figures/generation_metrics_table.png`：真实实验指标表或截图，例如 RMSE 或其他生成质量指标。不要填模拟数值。
- `figures/depth_error_map.png`：可选的深度误差图，用于展示生成结果与 reference/GT 的误差分布。

---

## Timing Summary

- Page 1: 15 seconds
- Page 2: 25 seconds
- Page 3: 30 seconds
- Page 4: 25 seconds
- Page 5: 35 seconds
- Page 6: 35 seconds
- Page 7: 45 seconds
- Page 8: 65 seconds
- Page 9: 25 seconds
- Total: 5 minutes
