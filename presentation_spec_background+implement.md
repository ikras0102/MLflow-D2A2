# D2A2 + MLOps 页面规划规格

## 使用约束

- 仅规划页面，不生成 PPT。
- 不生成图片，只描述建议视觉元素。
- 每页上屏文字控制在 80 字以内。
- 口播重点为讲者备注，不作为上屏文案。

## Page 1：背景 / 任务动机

- 标题：低质深度图到高质量重建
- 页面目标：说明深度重建需求与 GDSR 任务来源。
- 页面内容：
  - 低成本 depth 分辨率低
  - 噪声多，边界不清
  - RGB 含边缘结构
  - GDSR：LRD + RGB -> HRD
- 建议视觉元素：低质 depth、RGB guidance、HR depth 的三段箭头示意。
- 预计讲述时间：25 秒
- 口播重点：深度图提供几何信息，但低成本采集质量不足；RGB 更易获得且能引导重建。

## Page 2：背景 / 工程问题

- 标题：模型方案到工程需求
- 页面目标：从 D2A2 过渡到 MLOps 闭环。
- 页面内容：
  - D2A2：RGB + LRD + MDE
  - 常规流程：训 / 验 / 测 / 调参
  - 参数、数据、指标易分散
  - 引入 MLOps 实验闭环
- 建议视觉元素：D2A2 输入到传统流程，再到 MLOps 闭环的流程线。
- 预计讲述时间：30 秒
- 口播重点：D2A2 是核心模型方案；实验增多后需要统一管理参数、数据、checkpoint 和指标。

## Page 3：实现 / 总体架构

- 标题：科研训练流程工程化
- 页面目标：概览 D2A2 与 MLOps 工具分工。
- 页面内容：
  - D2A2 保持核心训练
  - Git 管代码版本
  - DVC 管数据快照
  - MLflow 管实验记录
  - Optuna 管调参搜索
- 建议视觉元素：代码、数据、训练、追踪、调参围绕 D2A2 的模块图。
- 预计讲述时间：30 秒
- 口播重点：工作重点不是重写算法，而是把科研实验流程组织成可追踪、可复现、可比较的工程闭环。

## Page 4：实现 / 训练验证测试

- 标题：统一训练验证测试
- 页面目标：说明单次实验如何闭环。
- 页面内容：
  - train 按 epoch 运行
  - validate 定期算 RMSE
  - 保存 val 最优权重
  - 加载最优权重再 test
  - 同一 run 记录最终 RMSE
- 建议视觉元素：train -> validate -> best checkpoint -> test -> MLflow run 流程图。
- 预计讲述时间：30 秒
- 口播重点：测试结果对应训练中验证集最优的 checkpoint，而不是手工指定或直接取最后一轮。

## Page 5：实现 / MLflow 追踪

- 标题：MLflow 实验主索引
- 页面目标：突出 MLflow 统一记录实验证据。
- 页面内容：
  - Params：lr、batch、loss
  - Metrics：loss、val RMSE、test RMSE
  - Artifacts：checkpoint、log
  - Tags：git、DVC、pipeline
- 建议视觉元素：MLflow run 卡片，分为 Params、Metrics、Artifacts、Tags 四栏。
- 预计讲述时间：50 秒
- 口播重点：MLflow 把参数、指标、日志、模型产物和数据版本集中到同一个 run，便于比较和复盘。

## Page 6：实现 / DVC 数据版本

- 标题：数据版本绑定结果
- 页面目标：说明结果可复现依赖明确数据快照。
- 页面内容：
  - NYUv2 由 .dvc 记录
  - hash / size / nfiles 入库
  - git_commit 写入 MLflow
  - 结果绑定代码与数据
- 建议视觉元素：datasets/NYUv2.dvc 指向 MLflow metadata 的连线。
- 预计讲述时间：35 秒
- 口播重点：实验结果不仅对应参数，也要对应代码 commit 和数据 hash，避免数据变化导致结果无法复现。

## Page 7：实现 / Optuna 调参

- 标题：Optuna 调参闭环
- 页面目标：说明多组参数如何被系统化比较。
- 页面内容：
  - 10 trials，每次 200 epochs
  - 目标：minimize RMSE
  - 搜索 lr / batch / loss / scheduler
  - trial 作为 nested run
  - parent 记录 best trial
- 建议视觉元素：parent run 下挂多个 trial run，并按 RMSE 排序。
- 预计讲述时间：45 秒
- 口播重点：Optuna 负责搜索，MLflow 负责记录；每个 trial 都有独立曲线、指标和 checkpoint。

## Page 8：实现 / 最优模型

- 标题：最优模型选择
- 页面目标：说明最终模型如何确定，并简要带过 PyFunc。
- 页面内容：
  - trial 内：val RMSE 选权重
  - trial 间：objective RMSE 选组
  - 候选：best_parameter
  - PyFunc：三输入推理封装
- 建议视觉元素：两层选择图：trial 内 checkpoint，trial 间 best trial。
- 预计讲述时间：20 秒
- 口播重点：最终模型来自 RMSE 最低 trial 中验证集表现最好的权重；PyFunc 只是服务化补充。

## 总体口播重点

- 背景：需求来自高质量深度重建，RGB 可作为低质 depth 的引导信息。
- 模型：D2A2 是核心训练对象，不是汇报唯一重点。
- 工程：MLflow、DVC、Optuna 共同形成可追踪、可复现、可比较的实验闭环。
- 选择：最终模型按验证集最优权重和跨 trial RMSE 两层确定。

## 时间估算

- 背景部分：55 秒
- 实现部分：3 分 30 秒
- 合计：约 4 分 25 秒
