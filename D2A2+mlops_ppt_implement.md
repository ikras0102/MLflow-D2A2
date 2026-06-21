# D2A2 + MLOps 汇报实现部分底稿

## 实现部分定位

实现部分建议控制在 2 分 30 秒左右，是整个 5 分钟汇报的主体。这里不要按代码文件逐个介绍，而是围绕“我们怎样参考一个已有科研算法，并把科研中常见的训练、验证、测试和调参流程工程化”来讲。

实现部分的主线可以概括为：

```text
D2A2 作为实验对象
    |
科研常用训练流程
    |
工程化实验闭环
    |
MLflow 实验追踪
    |
DVC 数据版本记录
    |
Optuna 超参数搜索
    |
最优模型选择
```

其中 MLflow 是重点，DVC 和 Optuna 分别说明数据可复现和参数调优能力，PyFunc 只简要带过。

## 第 1 页：总体实现架构

页面标题可以写：

```text
把科研训练流程工程化
```

页面主体建议放一张流程图：

```text
参考已有科研算法 D2A2
        |
常见科研实验流程
  - 配置参数
  - 准备数据
  - 训练模型
  - 验证选择 checkpoint
  - 测试评估
  - 多组参数对比
        |
工程化 MLOps 实现
  - Git: 代码版本
  - DVC: 数据版本
  - MLflow: 实验追踪
  - Optuna: 超参数搜索
        |
可复现、可比较的最优模型选择
```

口播底稿：

```text
实现部分的讲法可以从科研训练流程出发。我们不是重新提出一个新的深度超分辨率算法，而是参考已有的 D2A2 算法，把科研中常见的训练、验证、测试和多组参数对比过程，放进一个更工程化的 MLOps 流程里。

在普通科研实验中，我们通常会反复修改参数、切换数据、训练模型、保存 checkpoint，再手动比较验证集或测试集结果。这个过程可以跑通模型，但当实验数量增加时，参数、数据版本、checkpoint 和指标很容易分散在不同目录和日志里。

所以我们的实现重点是把这个流程系统化：Git 负责代码版本，DVC 负责数据版本，D2A2 仍然负责核心模型训练，MLflow 统一记录每一次实验的参数、指标、日志和模型产物，Optuna 负责组织多组超参数搜索。

这样汇报重点就不是“给现成算法加工具”，而是“把科研模型实验过程工程化”：每一次实验都可以追溯、比较和复现，最终模型选择也有完整证据链。
```

建议用时：30 秒。

## 第 2 页：训练-验证-测试一体化

页面标题可以写：

```text
统一 train -> validate -> test 流程
```

页面主体建议放流程：

```text
build model
    |
build train/test dataloader
    |
train each epoch
    |
validate by interval
    |
save best_parameter by val_mean_rmse
    |
load best_parameter
    |
test best model
    |
log best_model_test_rmse_for_gate
```

可以补充关键入口：

```text
单次实验入口：
train_test_d2a2_mlflow_dvc.py

调参实验入口：
train_test_d2a2_mlflow_optuna_dvc.py
```

口播底稿：

```text
第一步改造是把训练和测试合并成一个统一流程。

原始项目中训练和测试是两个相对独立的入口，训练保存 checkpoint，测试再手动指定 checkpoint 路径。这样很难保证某次测试结果一定对应某次训练的最优模型。

现在的流程是在训练过程中定期验证，用 val_mean_rmse 判断当前模型是否优于历史最好结果。如果更优，就保存 best_parameter。训练结束后，脚本会自动重新加载这个 best_parameter，再执行测试，并把测试结果写回同一个 MLflow run。

因此，一个 run 里既能看到训练过程，也能看到验证集最优 checkpoint 的最终测试表现。
```

建议用时：30 秒。

## 第 3 页：MLflow 实验追踪设计

页面标题可以写：

```text
MLflow 作为实验主索引
```

页面主体建议分四块：

```text
Params
- lr / batch_size / loss
- scale / dataset / input_size
- optimizer / scheduler / criterion
- dataset_dir / DVC metadata

Metrics
- train_running_loss
- train_epoch_loss
- val_mean_rmse
- best_mean_rmse
- best_model_test_rmse_for_gate

Artifacts
- best_parameter
- last_parameter
- train.log
- test.log
- dvc_metadata.json

Tags
- pipeline
- run_id
- git_commit
- dvc_status
```

口播底稿：

```text
这部分是本次实现的重点。我们把 MLflow 作为整个实验系统的主索引。

每次训练都会创建一个 MLflow run。run 里首先记录参数，包括学习率、batch size、loss、scale、数据集路径、优化器、scheduler 和模型配置。这样后面比较实验时，不需要回忆当时命令行传了什么参数。

其次记录指标。训练阶段记录 loss 和学习率变化；验证阶段记录 val_mean_rmse 和 best_mean_rmse；训练结束后加载最优 checkpoint 做测试，并记录 best_model_test_rmse_for_gate。这个指标就是后续选择模型时最重要的比较依据，RMSE 越低越好。

最后记录 artifacts，包括 best checkpoint、last checkpoint、训练日志、测试日志，以及 DVC 数据版本 metadata。也就是说，MLflow 不只是记录一个最终分数，而是把一次实验发生的主要证据都组织到同一个 run 下面。
```

建议用时：50 秒。

## 第 4 页：DVC 数据版本管理

页面标题可以写：

```text
DVC 让实验结果绑定到明确数据快照
```

页面主体建议放：

```text
datasets/NYUv2.dvc

md5:    cdca213631138298accea67d6ca26e0d.dir
size:   2063786418
nfiles: 4347
path:   NYUv2
```

再放 MLflow 中记录的字段：

```text
git_commit
git_dirty
dvc_dataset_path
dvc_dataset_hash
dvc_dataset_size
dvc_dataset_nfiles
dvc_status
```

口播底稿：

```text
深度超分辨率任务对数据非常敏感。只记录 dataset_dir 这个字符串是不够的，因为目录内容、MDE 伪深度、split 或预处理结果变化后，实验结果都会变。

所以我们引入 DVC 管理数据版本。当前 NYUv2 数据由 datasets/NYUv2.dvc 记录，它包含数据目录的 hash、大小和文件数。训练前可以通过 DVC 检查或拉取数据，训练时再把 DVC metadata 写入 MLflow。

这样一次实验结果不只对应一组超参数，还能追溯到明确的 Git commit 和 DVC dataset hash。也就是说，如果以后要复现实验，不只是找到 checkpoint，还要知道这个 checkpoint 是由哪份代码和哪份数据训练出来的。
```

建议用时：35 秒。

## 第 5 页：Optuna 超参数搜索

页面标题可以写：

```text
Optuna + MLflow 的调参闭环
```

页面主体建议放：

```text
Optuna study: minimize RMSE

trial_000 -> nested MLflow run -> objective_rmse
trial_001 -> nested MLflow run -> objective_rmse
...
trial_009 -> nested MLflow run -> objective_rmse

parent run:
- best_trial_number
- best_objective_rmse
- best_lr
- best_batch_size
- best_loss
- best_step_size
- best_lr_gamma
```

搜索空间建议具体列出：

```text
lr:         1e-5 到 1e-3，log scale
batch_size: 2 / 4 / 8
loss:       l1 / mask
step_size:  10 / 20 / 30
lr_gamma:   0.3 / 0.5 / 0.7
```

口播底稿：

```text
在调参部分，我们用 Optuna 管理超参数搜索，并且把它和 MLflow 结合起来。

当前实验设计是 10 个 trial，每个 trial 训练 200 epoch。Optuna 的目标是 minimize RMSE。每一个 trial 都会作为一个 MLflow nested run，因此每组参数都有自己独立的训练曲线、验证 RMSE、测试 RMSE 和 checkpoint。

当前搜索空间包括五个参数：学习率、batch size、loss 类型、StepLR 的 step size，以及学习率衰减系数 lr_gamma。学习率使用 log scale 在 1e-5 到 1e-3 之间搜索，batch size 在 2、4、8 中选择，loss 在 l1 和 mask loss 中选择。

调参结束后，parent run 会记录 best trial 编号、best objective RMSE，以及最优参数。这样我们可以直接在 MLflow UI 里按 objective_rmse 或 best_model_test_rmse_for_gate 排序，找到最优 trial 和对应 checkpoint。
```

建议用时：45 秒。

## 第 6 页：最优模型选择与服务化补充

页面标题可以写：

```text
从 best trial 到 best model
```

页面主体建议放：

```text
within each trial:
val_mean_rmse -> best_parameter

across trials:
objective_rmse -> best_trial

final candidate:
best trial's checkpoints/best_parameter
```

PyFunc 只保留一行：

```text
PyFunc wrapper: 解决 D2A2 三输入模型的服务化调用问题
```

口播底稿：

```text
最后，最优模型不是直接取最后一个 epoch 的 checkpoint，而是分两层选择。

在单个 trial 内部，我们根据验证集 val_mean_rmse 保存 best_parameter。训练结束后，加载这个 best_parameter 做测试，得到 best_model_test_rmse_for_gate。

在多个 trial 之间，我们比较 objective_rmse，选择 RMSE 最低的 trial。最终候选模型就是这个 best trial 对应的 best_parameter。

服务化部分我们也做了 PyFunc 封装。因为 D2A2 的输入不是单个 tensor，而是 RGB、LR depth 和 MDE 三路输入，所以 PyFunc 主要用于把三输入推理过程包装成 MLflow serving 可以调用的接口。不过它不是寻找最优模型的核心，所以汇报里只作为扩展说明。
```

建议用时：20 秒。

## 实现部分总口播精简版

如果现场时间紧，可以把实现部分压缩为下面这段：

```text
实现上，我们是以 D2A2 这个已有科研算法为实验对象，把科研中常见的训练、验证、测试和多组参数对比流程，工程化成一个 MLOps 实验闭环。

首先，我们保留 D2A2 的核心模型和数据流程，但把实验流程重新组织起来。训练过程中定期验证，用 val_mean_rmse 保存 best_parameter；训练结束后自动加载这个 best checkpoint 做测试，并把 best_model_test_rmse_for_gate 作为统一的模型选择指标。

其次，我们把 MLflow 作为实验追踪主线。每次实验都是一个 MLflow run，里面记录超参数、模型配置、训练 loss、验证 RMSE、测试 RMSE、checkpoint、日志和数据版本 metadata。这样科研实验中原本分散在命令行、本地目录和日志文件里的信息，被统一组织成可以查询和比较的实验记录。

然后，我们用 DVC 管理数据版本。当前 NYUv2 数据由 datasets/NYUv2.dvc 记录，里面有数据 hash、大小和文件数。训练时会把 git_commit、dvc_dataset_hash、dvc_dataset_size、dvc_dataset_nfiles 等字段写入 MLflow，所以实验结果不只对应一组参数，还能追溯到具体代码和具体数据快照。

最后，调参部分用 Optuna 实现。当前实验是 10 个 trial，每个 trial 训练 200 epoch，优化目标是 minimize RMSE。搜索空间包括 lr、batch_size、loss、step_size 和 lr_gamma。每个 trial 都是一个 MLflow nested run，调参结束后 parent run 记录最优 trial 和最优参数。

因此，这个实现的重点不是重新设计一个算法模块，而是把科研模型实验过程变得可追踪、可复现、可比较。最终模型选择也不再依赖手工判断最后一个 checkpoint，而是通过 MLflow 和 Optuna 找到 RMSE 最低的 trial，再取该 trial 中验证集表现最好的 best_parameter。
```

## 实现部分强调点

汇报时建议反复强调三点：

```text
1. 工作对象是科研训练流程：以 D2A2 为载体，把训练、验证、测试、调参变成工程化闭环。

2. MLflow 是核心：它把参数、指标、模型产物、日志和数据版本统一到一个 run。

3. DVC 和 Optuna 分别补齐可复现和可调参：前者绑定数据 hash，后者让每个 trial 都有完整记录并可按 RMSE 比较。
```

不建议在实现部分花太多时间讲：

```text
1. D2A2 论文细节。
2. PyFunc REST 请求细节。
3. 每个 Python 文件的完整代码结构。
4. 原始项目所有历史问题。
```

这些内容可以放到背景、实验或答辩问答中补充。实现部分最重要的是讲清楚：我们不是孤立展示几个工具，而是把 MLflow、DVC 和 Optuna 嵌入科研训练流程，使实验从“能跑出结果”进一步变成“结果可追踪、过程可复现、参数可比较”。
