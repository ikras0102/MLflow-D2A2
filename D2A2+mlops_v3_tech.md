# D2A2 + MLOps 技术报告 v3（技术版）

## 0. 文档定位

本文档面向课程报告或 PPT 技术部分，不展开 D2A2 的论文背景和任务背景，而是从工程架构角度重新组织当前仓库的 MLOps 改造结果。

本文重点说明四件事：

- 如何把 D2A2 从单次本地训练脚本，组织成可追踪的训练、验证、测试闭环。
- 如何以 MLflow 作为实验主线，统一记录参数、指标、checkpoint、测试结果和模型产物。
- 如何用 DVC 管理数据版本，并把 Git/DVC 元数据写入 MLflow run。
- 如何用 Optuna 对关键超参数做系统搜索，并把每个 trial 纳入 MLflow 对比。

PyFunc 封装只作为模型服务化补充说明。它解决的是三输入推理接口如何对外暴露，不是本次“得到最优模型”的核心环节。

## 1. 技术架构总览

当前项目可以理解为一个围绕 D2A2 模型构建的实验工程系统，而不只是一个训练脚本。核心思想是：

```text
Git 管代码
    |
DVC 管数据
    |
D2A2 训练 / 验证 / 测试
    |
MLflow 管实验记录和模型产物
    |
Optuna 管超参数搜索
    |
PyFunc 管可选服务化封装
```

从架构层面拆分，系统可以分成六层：

```text
+-----------------------------+
| 6. Serving Layer             |
|    MLflow PyFunc wrapper     |
+-----------------------------+
| 5. HPO Layer                 |
|    Optuna study / trial      |
+-----------------------------+
| 4. Tracking Layer            |
|    MLflow params / metrics   |
|    artifacts / checkpoints   |
+-----------------------------+
| 3. Training Layer            |
|    model / dataloader        |
|    optimizer / scheduler     |
|    loss / validate / test    |
+-----------------------------+
| 2. Data Version Layer        |
|    DVC .dvc pointer          |
|    DVC cache / remote        |
+-----------------------------+
| 1. Code & Config Layer       |
|    Git commit / option.py    |
|    train_test entrypoints    |
+-----------------------------+
```

在这个架构中，MLflow 是贯穿训练过程的中心。DVC 和 Git 负责回答“这次实验用的代码和数据是什么”，Optuna 负责回答“为什么选择这一组参数”，MLflow 负责把这些信息绑定成一个可查询、可比较、可复现的实验记录。

## 2. D2A2 训练任务在工程中的位置

D2A2 的模型调用不是单输入，而是三输入：

```python
model(guidance, target, mde)
```

三路输入含义如下：

```text
guidance: HR RGB guidance image
target:   LR depth map
mde:      monocular depth estimation pseudo-depth
```

训练入口根据 `--scale` 选择不同模型文件：

```text
scale = 4   -> models/D2A2_depthanything.py
scale = 8   -> models/D2A2_depthanything_L_scale8.py
scale = 16  -> models/D2A2_depthanything_L_scale16.py
```

数据加载层根据 `--dataset` 和 `--dataset_dir` 构建 dataloader。训练目前主要面向 NYUv2：

```text
NYU_v2_dataset(
    root_dir=args.dataset_dir,
    scale=args.scale,
    augment=args.augment,
    input_size=args.input_size,
)
```

测试阶段支持：

```text
nyu / lu / middlebury / rgbdd
```

模型优化部分由三类对象组成：

```text
optimizer: Adam
scheduler: StepLR(step_size=args.step_size, gamma=args.lr_gamma)
criterion: L1Loss 或 MaskLoss
```

其中 `MaskLoss` 会根据深度图经过 bicubic 上采样再下采样后的差异构造权重，使损失对深度结构变化更敏感；`L1Loss` 则是直接的像素级绝对误差。

评价指标使用 RMSE。计算时会先根据样本记录的 `min` 和 `max` 把归一化深度还原到原深度范围；NYUv2 测试时还会裁掉边界 6 个像素后再计算 RMSE。

## 3. 标准训练-测试闭环

当前更适合作为标准单次实验入口的是：

```text
train_test_d2a2_mlflow_dvc.py
```

它把原来分离的训练和测试串成一个完整闭环：

```text
setup_seed
    |
prepare_dvc_data
    |
mlflow.start_run
    |
build_model
    |
build_train_loader / build_test_loader
    |
build_optimizer_scheduler / build_criterion
    |
MLFlowTrainer.train
    |
save best_parameter by val_mean_rmse
    |
load best_parameter
    |
Tester.validate on best model
    |
log best_model_test_rmse_for_gate
```

这个闭环解决了原始研究代码中的一个关键问题：训练和测试不再是两个彼此独立的本地过程，而是被绑定进同一个 MLflow run。也就是说，一个 run 可以同时回答：

- 使用了哪一份代码、哪一份数据、哪一组参数。
- 训练过程中的 loss 和验证 RMSE 如何变化。
- 哪个 epoch 得到了验证集上的 best checkpoint。
- best checkpoint 在测试集上的 RMSE 是多少。
- checkpoint、日志、测试输出保存在哪里。

其中最重要的统一比较指标是：

```text
best_model_test_rmse_for_gate
```

它表示“训练结束后加载验证集最优 checkpoint，再执行测试得到的平均 RMSE”。在当前作业场景中，这个指标可以作为模型选择和后续发布 gate 的主指标，数值越低越好。

## 4. MLflow：实验追踪主线

本次作业最关键的改造是把 MLflow 引入 D2A2 项目。MLflow 在这里不是附加日志工具，而是实验系统的主索引。

### 4.1 Run 组织方式

单次训练-测试使用一个 run：

```text
experiment: MLflow+D2A2
tag pipeline = dvc_train_then_test_best_model
```

Optuna 调参使用 parent run + nested trial run：

```text
parent run:
    tag pipeline = dvc_optuna_hpo_parent
    records best_trial_number / best_objective_rmse / best_* params

nested trial run:
    tag pipeline = optuna_train_then_test_best_model
    tag trial_number = <trial.number>
    records this trial's params / metrics / artifacts
```

这种组织方式适合课程报告展示：parent run 表示一次调参任务，nested run 表示每一次候选参数实验。这样既能看整体最优结果，也能展开比较每个 trial 的训练曲线和模型产物。

### 4.2 记录的参数

`MLFlowTrainer.train()` 会记录 `option.py` 中的主要参数，例如：

```text
model_name
model_file
scale
dataset
dataset_dir
pretrain_path
n_resblocks
n_feats
res_scale
input_size
epoch
batch_size
augment
lr
step_size
loss
validate_interval
mlflow_experiment
mlflow_run_name
cuda_devices
lr_gamma
n_trials
log_interval
save_last_checkpoint
save_all_epochs
save_mlflow_model
save_best_checkpoint_artifact
dvc_pull
dvc_required
dvc_remote
dvc_jobs
dvc_data_paths
```

训练代码还会额外记录一些派生参数：

```text
optimizer
scheduler
criterion
model_class
train_dataset_size
test_dataset_size
train_result_root
tested_checkpoint_path
best_parameter_path
best_epoch
```

这些参数的作用是让一次实验脱离终端命令之后仍然可解释。例如只看 MLflow UI，就可以知道这次 run 用的是哪个 scale、哪种 loss、多少 batch size、哪个 checkpoint 被测试。

### 4.3 记录的指标

训练阶段按 step 和 epoch 记录：

```text
train_running_loss
train_batch_loss
train_epoch_loss
optimizer_lr
```

验证阶段记录：

```text
val_mean_rmse
best_mean_rmse
best_epoch
```

测试阶段记录：

```text
best_model_test_mean_rmse
best_model_test_total_time
best_model_test_avg_time
best_model_test_rmse_for_gate
```

Optuna trial 中还会记录：

```text
objective_rmse
```

其中 `objective_rmse` 与 `best_model_test_rmse_for_gate` 在当前实现中指向同一个目标：best checkpoint 测试后的平均 RMSE。Optuna 根据这个值做 minimize。

### 4.4 记录的产物

MLflow artifacts 中包含：

```text
option.py
logs/train.log
best_model_test/test.log
checkpoints/best_parameter
checkpoints/last_parameter
metadata/dvc_metadata.json
best_model_test/*
```

其中：

- `best_parameter` 是验证 RMSE 最优时保存的 checkpoint。
- `last_parameter` 是训练最后阶段保存的 checkpoint。
- `dvc_metadata.json` 保存 Git 和 DVC 数据版本信息。
- `best_model_test` 保存测试日志，以及在 `--save` 打开时保存的深度图、热力图等输出。

如果启用：

```text
--save_mlflow_model
```

还可以额外记录：

```text
model
best_model
```

对于课程报告来说，需要突出的是：MLflow 让“参数、指标、模型文件、数据版本、日志”不再分散在本地目录里，而是被组织进同一个 run。

## 5. DVC：数据版本管理具体化

深度超分辨率任务对数据版本非常敏感。同样的模型参数，在不同数据目录、不同 MDE 伪深度、不同 split 或不同预处理结果上，RMSE 都可能不同。因此，只在 MLflow 中记录 `dataset_dir` 字符串是不够的，还需要记录数据快照本身。

DVC 在当前项目中的职责是：

```text
Git:
    记录代码和 .dvc 指针文件

DVC:
    记录大数据目录的内容 hash
    管理本地 cache 或远程 storage

MLflow:
    记录本次 run 对应的 Git commit 和 DVC hash
```

### 5.1 当前数据指针

当前仓库已有 DVC 指针：

```text
datasets/NYUv2.dvc
```

内容摘要如下：

```yaml
outs:
- md5: cdca213631138298accea67d6ca26e0d.dir
  size: 2063786418
  nfiles: 4347
  path: NYUv2
```

这里的 `path: NYUv2` 是相对 `datasets/NYUv2.dvc` 所在目录解析的，因此对应的数据目录是：

```text
datasets/NYUv2
```

这条指针说明当前 NYUv2 数据快照具备以下版本信息：

```text
DVC directory hash: cdca213631138298accea67d6ca26e0d.dir
size:               2063786418 bytes
nfiles:             4347
path:               datasets/NYUv2
```

这些信息会成为实验可复现性的基础：只要 Git commit 和 DVC hash 一致，理论上就能还原到同一份训练数据。

### 5.2 训练前的数据准备

DVC 相关入口参数包括：

```text
--dvc_pull
--dvc_required
--dvc_remote
--dvc_jobs
--dvc_data_paths
--skip_dvc_metadata
```

`prepare_dvc_data(args)` 的逻辑可以概括为：

```text
1. 从 args.dataset_dir 收集主数据路径。
2. 从 args.dvc_data_paths 收集额外数据路径。
3. 归一化路径并去重。
4. 如果传入 --dvc_pull，则执行 dvc pull。
5. 如果指定 --dvc_remote，则 dvc pull -r <remote>。
6. 如果指定 --dvc_jobs，则 dvc pull -j <jobs>。
7. 检查所有数据路径是否存在。
8. 如果传入 --dvc_required，则执行 dvc status。
```

典型命令形式可以写成：

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2 \
  --dvc_pull \
  --dvc_required
```

如果一次训练还依赖额外 DVC 数据，例如 MDE 伪深度、预训练权重或预处理缓存，可以通过：

```bash
--dvc_data_paths MDE,pretrained
```

把它们一起纳入 DVC 元数据记录。

### 5.3 写入 MLflow 的 DVC 元数据

`log_dvc_metadata(args)` 会把 Git/DVC 信息写入当前 MLflow run。主要字段包括：

```text
git_commit
git_dirty
dvc_available
dvc_version
dvc_dataset_path
dvc_dataset_dvc_file
dvc_dataset_hash
dvc_dataset_size
dvc_dataset_nfiles
dvc_data_0_path
dvc_data_0_exists
dvc_data_0_dvc_file
dvc_data_0_hash
dvc_data_0_size
dvc_data_0_nfiles
dvc_status
```

如果有多个数据路径，还会继续记录：

```text
dvc_data_1_*
dvc_data_2_*
...
```

同时，完整元数据会被保存为 artifact：

```text
metadata/dvc_metadata.json
```

因此，MLflow run 中的数据版本不是一个口头描述，而是可以落到具体 hash、具体 `.dvc` 文件、具体 Git commit 的结构化信息。

### 5.4 DVC 在报告中的表述重点

报告中建议这样表达 DVC 的价值：

```text
MLflow 记录实验结果，DVC 记录数据内容版本。
二者结合后，一次 RMSE 结果不只对应一组参数，
还对应一个明确的 Git commit 和一个明确的数据 hash。
```

这比单独保存本地 checkpoint 更可靠，因为 checkpoint 只能说明模型权重是什么，不能说明它由哪份数据训练出来。

## 6. Optuna：超参数调优具体化

当前标准调参入口是：

```text
train_test_d2a2_mlflow_optuna_dvc.py
```

它在单次训练-测试闭环外层增加 Optuna study，并继续保留 DVC 和 MLflow 记录能力。

整体流程如下：

```text
prepare_dvc_data
    |
MLflow parent run
    |
Optuna study(direction="minimize")
    |
trial 0 -> nested MLflow run -> train -> test -> objective_rmse
trial 1 -> nested MLflow run -> train -> test -> objective_rmse
...
    |
study.best_trial
    |
parent run logs best params and best RMSE
```

### 6.1 调优目标

Optuna 的优化方向是：

```text
direction = minimize
```

目标函数返回：

```text
objective_rmse = best_model_test_rmse_for_gate
```

也就是每个 trial 内部完成训练后，加载验证集最优 checkpoint，再在测试 loader 上计算平均 RMSE。RMSE 越低，该 trial 越优。

### 6.2 当前搜索空间

当前实现调优 5 个参数。

第一个是学习率：

```python
trial.suggest_float("lr", 1e-5, 1e-3, log=True)
```

含义：

```text
搜索范围: 0.00001 到 0.001
采样方式: log scale
作用: 控制 Adam 参数更新步长，是最影响收敛稳定性的参数之一
```

第二个是 batch size：

```python
trial.suggest_categorical("batch_size", [2, 4, 8])
```

含义：

```text
候选值: 2 / 4 / 8
作用: 影响显存占用、梯度估计稳定性和训练速度
```

第三个是 loss：

```python
trial.suggest_categorical("loss", ["l1", "mask"])
```

含义：

```text
l1:   使用 torch.nn.L1Loss
mask: 使用 MaskLoss
作用: 比较普通像素误差和结构加权误差对 RMSE 的影响
```

第四个是 StepLR 的 step size：

```python
trial.suggest_categorical("step_size", [10, 20, 30])
```

含义：

```text
候选值: 10 / 20 / 30
作用: 控制学习率衰减间隔
```

第五个是学习率衰减系数：

```python
trial.suggest_categorical("lr_gamma", [0.3, 0.5, 0.7])
```

含义：

```text
候选值: 0.3 / 0.5 / 0.7
作用: 控制每次 StepLR 衰减时学习率乘上的比例
```

这 5 个参数覆盖了“优化器行为、batch 统计稳定性、损失函数、学习率调度”四类关键因素，适合作为课程作业中的第一版调参空间。`n_resblocks`、`n_feats` 等结构参数会显著增加训练成本，当前没有纳入搜索空间。

### 6.3 固定但需要记录的参数

除了被 Optuna 搜索的参数，每个 trial 还会记录：

```text
scale
dataset
input_size
augment
epoch
validate_interval
```

每个 trial 使用：

```text
setup_seed(20 + trial.number)
```

这样不同 trial 有不同随机种子，避免所有 trial 完全绑定同一个随机状态。

### 6.4 Optuna 与 MLflow 的结合方式

每个 trial 都是一个 nested MLflow run，并记录：

```text
lr
batch_size
loss
step_size
lr_gamma
scale
dataset
input_size
augment
objective_rmse
best_model_test_rmse_for_gate
checkpoints/best_parameter
logs/train.log
best_model_test/test.log
```

调参结束后，parent run 记录：

```text
best_trial_number
best_objective_rmse
best_lr
best_batch_size
best_loss
best_step_size
best_lr_gamma
```

因此，最终选择最优模型时，不需要手工翻日志，而是可以直接在 MLflow UI 中按 `objective_rmse` 或 `best_model_test_rmse_for_gate` 排序，定位最优 trial，再取该 trial 的 `checkpoints/best_parameter` 作为候选最优模型。

### 6.5 方法论说明

当前工程中把 best checkpoint 的测试 RMSE 作为 Optuna objective，适合作业中演示“调参产生最优模型”的闭环。如果按严格机器学习实验规范，建议保留独立 test set，只用 validation RMSE 做 HPO objective，最后仅对选出的模型做一次 test。这个点可以作为报告中的后续优化方向说明。

## 7. 最优模型产生链路

综合 MLflow、DVC 和 Optuna 后，最优模型不是由某个本地文件名决定的，而是由一条完整证据链决定：

```text
Git commit
    |
DVC dataset hash
    |
MLflow parent run
    |
Optuna best trial
    |
nested MLflow run
    |
best validation checkpoint
    |
best_model_test_rmse_for_gate
```

单个 trial 内部的模型选择逻辑是：

```text
训练过程中定期 validate
    |
如果 val_mean_rmse 低于历史 best_mean_rmse
    |
保存 best_parameter
    |
训练结束后加载 best_parameter
    |
执行 test
    |
记录 best_model_test_rmse_for_gate / objective_rmse
```

多个 trial 之间的模型选择逻辑是：

```text
比较所有 nested run 的 objective_rmse
    |
选择 RMSE 最低的 trial
    |
读取该 trial 的 best params
    |
使用该 trial 的 checkpoints/best_parameter 作为最优候选模型
```

这种链路比“最后一个 checkpoint”更合理，因为它同时考虑了验证集表现、测试指标、超参数配置和数据版本。

## 8. PyFunc 服务化简述

D2A2 的模型接口是三输入：

```text
guidance / target / mde
```

直接使用 `mlflow.pytorch.log_model` 虽然可以保存 PyTorch 模型，但默认 REST 请求形式不适合表达这三个 tensor。因此项目中增加了：

```text
mlflow_transform2pyfunc_model.py
```

其中 `D2A2ServingWrapper` 继承：

```python
mlflow.pyfunc.PythonModel
```

`predict()` 接收一个 dataframe，每行包含：

```text
rgb_path
depth_path
mde_path
```

wrapper 在预测时加载三个 `.pt` tensor，补 batch 维度，移动到 GPU 或 CPU，然后调用 D2A2 模型完成推理。

这部分的作用是把“研究模型 checkpoint”进一步包装成“可被 MLflow serving 调用的模型接口”。但它位于最优模型产生之后，不参与训练、验证、调参和模型选择。因此在课程报告主线中可以简写，作为服务化扩展即可。

## 9. 工作沿革简述

项目改造可以简要概括为：

```text
原始 D2A2:
    train_d2a2.py / test_d2a2.py
    本地日志和本地 checkpoint 为主

MLflow 初版:
    train_d2a2_mlflow.py / test_d2a2_mlflow.py
    开始记录参数、指标和 artifact

Train + Test 一体化:
    train_test_d2a2_mlflow.py
    把训练 best checkpoint 和测试结果绑定到同一个 run

Optuna 调参:
    train_test_d2a2_mlflow_optuna.py
    parent run + nested trial run

DVC 数据版本:
    train_test_d2a2_mlflow_dvc.py
    train_test_d2a2_mlflow_optuna_dvc.py
    训练前检查数据，MLflow 中记录 Git/DVC 元数据

PyFunc 封装:
    mlflow_transform2pyfunc_model.py
    解决三输入推理接口的服务化表达
```

报告中不建议把这部分作为主体。更合适的写法是：项目最终形成了以 MLflow 为中心、DVC 和 Optuna 分别补足数据版本与参数搜索能力的实验架构。

## 10. PPT 展示建议

如果拆成 PPT，可以按以下页面组织：

```text
1. 技术目标：把 D2A2 改造成可追踪、可复现、可调参的实验系统
2. 总体架构：Git + DVC + D2A2 + MLflow + Optuna
3. 训练闭环：train -> validate -> best checkpoint -> test
4. MLflow 设计：run、params、metrics、artifacts
5. MLflow 指标：loss / val RMSE / best test RMSE / gate metric
6. DVC 数据版本：.dvc 指针、hash、size、nfiles、Git commit
7. DVC 与 MLflow 结合：把数据版本写入 run
8. Optuna 调参：搜索空间与 objective_rmse
9. 最优模型证据链：best trial -> best checkpoint -> RMSE
10. 服务化扩展：PyFunc 三输入 wrapper
```

其中第 4 页到第 8 页应作为重点，因为这部分最能体现 MLOps 工具真正进入了模型训练流程，而不只是作为外部说明文档存在。

## 11. 当前技术边界与后续改进

当前版本已经完成了课程作业所需的核心闭环：

```text
DVC 数据版本
    +
MLflow 实验追踪
    +
Optuna 超参数搜索
    +
best checkpoint 测试指标
```

仍可继续改进的点包括：

- 将 Optuna objective 从测试 RMSE 调整为验证 RMSE，保留独立 test set 做最终评估。
- 把 `save_mlflow_model` 作为标准实验配置的一部分，使最优模型不仅有 checkpoint，也有 MLflow model artifact。
- 增加 MLflow Model Registry 或简单发布脚本，基于 `best_model_test_rmse_for_gate` 自动判断是否注册新模型。
- 固化更多 DVC 管理对象，例如 MDE 伪深度目录、预训练权重、数据 split 文件和预处理缓存。
- 为 Optuna 使用持久化 storage，方便中断恢复和多人共享调参结果。

整体而言，当前项目的技术价值不只在于训练了一个 D2A2 模型，而在于把深度超分辨率实验放进了一个可审计的 MLOps 流程：每个候选模型都能追溯参数、指标、数据版本和产物，每次调参都能在 MLflow 中复盘和比较。
