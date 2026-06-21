# D2A2 + MLOps 技术报告 v1

## 1. 报告范围

本文档基于当前仓库实际状态，梳理 D2A2 从原始训练/测试脚本演进到 MLflow、Optuna、DVC 数据版本控制和 PyFunc 服务化的技术链路。

本文只覆盖技术实现与工程流程，不讨论业务背景、组织流程或非技术管理内容。

当前仓库中同时存在多代代码：

```text
原始 D2A2 代码
    |
    +-- train_d2a2.py
    +-- test_d2a2.py
    +-- utils.py
    |
    v
MLflow 初版
    |
    +-- train_d2a2_mlflow.py
    +-- test_d2a2_mlflow.py
    +-- train_test_d2a2_mlflow.py
    +-- mlflow_utils.py
    |
    v
Optuna + MLflow
    |
    +-- train_test_d2a2_mlflow_optuna.py
    |
    v
DVC + MLflow + Optuna
    |
    +-- dvc_training_utils.py
    +-- train_test_d2a2_mlflow_dvc.py
    +-- train_test_d2a2_mlflow_optuna_dvc.py
    |
    v
PyFunc serving
    |
    +-- mlflow_transform2pyfunc_model.py
    +-- test_d2a2_pyfunc_nyu.py
```

推荐把 `train_test_d2a2_mlflow_dvc.py` 作为当前标准单次训练入口，把 `train_test_d2a2_mlflow_optuna_dvc.py` 作为当前标准调参入口。

## 2. 当前仓库状态概览

### 2.1 核心模型与数据集

模型文件：

```text
models/D2A2_depthanything.py
models/D2A2_depthanything_L_scale8.py
models/D2A2_depthanything_L_scale16.py
```

数据集加载文件：

```text
datasets/nyu.py
datasets/lu.py
datasets/middlebury.py
datasets/rgbdd.py
datasets/common.py
```

D2A2 模型的关键输入是三路 tensor：

```python
model(guidance, target, mde)
```

含义：

```text
guidance: HR RGB guidance image
target:   LR depth map after downsample / normalization
mde:      monocular depth estimation pseudo-depth
```

NYUv2 数据目录结构要求：

```text
<dataset_dir>
    |
    +-- RGB/
    |   +-- 0.jpg
    |   +-- ...
    |
    +-- Depth/
    |   +-- 0.npy
    |   +-- ...
    |
    +-- MDE_relative/
        +-- 0.png
        +-- ...
```

训练时通过 `--dataset_dir` 显式指定该目录即可。

### 2.2 DVC 当前数据指针

当前已有 DVC 文件：

```text
datasets/NYUv2.dvc
```

内容摘要：

```yaml
outs:
- md5: cdca213631138298accea67d6ca26e0d.dir
  size: 2063786418
  nfiles: 4347
  path: NYUv2
```

其中 `path` 字段是相对于 `.dvc` 文件所在目录的路径。训练脚本本身不依赖固定目录名，只依赖命令行传入的 `--dataset_dir`。

### 2.3 环境兼容性

当前环境核心版本：

```text
python==3.7.16
torch==1.8.0
torchvision==0.9.0
mlflow==1.30.0
optuna==3.0.0
dvc==2.10.2
fsspec==2022.2.0
```

DVC 相关兼容性结论：

```text
dvc==2.10.2 + fsspec==2023.1.0  不兼容
dvc==2.10.2 + fsspec==2022.2.0  可用
```

曾出现的问题：

```text
ImportError: cannot import name 'fsspec_loop' from 'fsspec.asyn'
```

原因是 DVC 2.10.2 依赖旧版 `fsspec.asyn.fsspec_loop` 接口，而 `fsspec==2023.1.0` 中该接口已经变化。

当前 `environment.yaml` 已将 `fsspec` pin 到 `2022.2.0`。

## 3. 多版本代码职责划分

### 3.1 原始训练入口

文件：

```text
train_d2a2.py
utils.py
```

职责：

```text
build model
build NYUv2 train/test loader
train
periodic validation
save last_parameter / best_parameter
write local train.log
```

原始链路：

```text
option.py
    |
train_d2a2.py
    |
utils.Trainer
    |
result/trainresult/<run-dir>/
    |
    +-- train.log
    +-- best_parameter
    +-- last_parameter
```

遗留问题：

- CUDA 设备在脚本里硬编码：`CUDA_VISIBLE_DEVICES="0,1"`。
- 训练和测试是两个入口，best checkpoint 测试结果不天然属于同一个实验记录。
- 结果主要写入本地目录，没有结构化实验追踪。
- `utils.py` 使用 `logging.basicConfig()`，多次创建 trainer/tester 时日志容易串写或失效。
- `train_d2a2.py` 中存在 `args.loss is 'L1'` 这种 identity comparison 写法，语义上应使用 `==` 或 `.lower()`。
- `scheduler.step()` 在 batch 开头调用，可能引入 PyTorch scheduler 顺序警告或学习率行为不符合预期。

因此，原始入口适合保留用于对照，不建议作为新的标准实验入口。

### 3.2 原始测试入口

文件：

```text
test_d2a2.py
```

职责：

```text
load args.net_path
build model
load checkpoint
build test loader
Tester.validate()
write local test result
```

支持数据集：

```text
nyu
lu
middlebury
rgbdd
```

局限：

- 只依赖本地 checkpoint 路径。
- 不记录 MLflow run。
- 不自动关联训练时的参数、数据版本和 checkpoint 来源。

### 3.3 MLflow 初版入口

文件：

```text
train_d2a2_mlflow.py
test_d2a2_mlflow.py
mlflow_utils.py
```

改造目的：

```text
utils.Trainer / Tester
        |
        v
MLFlowTrainer / Tester
        |
        v
mlflow.log_params / log_metrics / log_artifact
```

这个阶段把训练过程中的 loss、validation RMSE、checkpoint、日志等写入 MLflow，但训练和测试仍然不是最理想的一体化流程。

### 3.4 MLflow train + test 一体化入口

文件：

```text
train_test_d2a2_mlflow.py
```

这是 MLflow 版本中更推荐的基础入口。

核心流程：

```text
build_model()
    |
build_train_loader()
    |
build_test_loader()
    |
MLFlowTrainer.train()
    |
best_parameter
    |
load_state_dict_flexible()
    |
Tester.validate(log_mlflow=True)
    |
best_model_test_rmse_for_gate
```

解决的问题：

- train 和 best model test 进入同一个 MLflow run。
- 训练验证集上的 best checkpoint 会被加载后再跑 test。
- 产生统一发布 gate 指标：`best_model_test_rmse_for_gate`。
- 兼容 `DataParallel` checkpoint 的 `module.` 前缀。

推荐基础 smoke test：

```bash
python train_test_d2a2_mlflow.py \
  --dataset nyu \
  --dataset_dir <dataset_dir> \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --cuda_devices 0 \
  --mlflow_run_name d2a2-mlflow-smoke
```

### 3.5 Optuna + MLflow 入口

文件：

```text
train_test_d2a2_mlflow_optuna.py
```

设计：

```text
parent MLflow run: optuna_hpo
    |
    +-- nested run: trial_000
    |       |
    |       +-- train
    |       +-- load best checkpoint
    |       +-- test
    |       +-- objective_rmse
    |
    +-- nested run: trial_001
    |
    v
best_trial / best_params / best_objective_rmse
```

当前搜索空间：

```text
lr:         1e-5 ~ 1e-3, log scale
batch_size: [2, 4, 8]
loss:       ["l1", "mask"]
step_size:  [10, 20, 30]
lr_gamma:   [0.3, 0.5, 0.7]
```

目标函数：

```text
minimize objective_rmse
```

其中 `objective_rmse` 来自 best checkpoint 在 test loader 上的 mean RMSE。

### 3.6 DVC 训练入口

文件：

```text
dvc_training_utils.py
train_test_d2a2_mlflow_dvc.py
train_test_d2a2_mlflow_optuna_dvc.py
```

DVC helper 职责：

```text
parse args.dataset_dir + args.dvc_data_paths
    |
check dvc executable
    |
optional dvc pull
    |
check data paths exist
    |
optional dvc status
```

参数：

```text
--dvc_pull       训练前执行 dvc pull
--dvc_required   DVC 不可用或数据缺失时提前失败
--dvc_remote     指定 dvc pull 的 remote
--dvc_jobs       指定 dvc pull 并发数
--dvc_data_paths 额外记录/检查的数据目录，逗号分隔
```

DVC + MLflow 单次训练推荐命令：

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir <dataset_dir> \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --cuda_devices 0 \
  --dvc_required \
  --mlflow_run_name d2a2-dvc-smoke
```

DVC + Optuna 推荐命令：

```bash
python train_test_d2a2_mlflow_optuna_dvc.py \
  --dataset nyu \
  --dataset_dir <dataset_dir> \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --n_trials 1 \
  --cuda_devices 0 \
  --dvc_required \
  --mlflow_run_name d2a2-dvc-optuna-smoke
```

## 4. MLflow 技术实现

### 4.1 `mlflow_utils.py` 核心对象

`mlflow_utils.py` 是 MLOps 化后的关键工具文件。

主要对象和函数：

```text
setup_seed()
calc_rmse()
load_state_dict_flexible()
log_dvc_metadata()
MLFlowTrainer
Tester
```

### 4.2 MLFlowTrainer

`MLFlowTrainer.train()` 负责：

```text
log args
log optimizer/scheduler/criterion/model info
log DVC metadata
train by epoch
log train_running_loss / train_epoch_loss / optimizer_lr
periodic validation
log val_mean_rmse
save best_parameter / last_parameter
log checkpoints and logs
return train_summary
```

返回结构：

```python
{
    "best_rmse": ...,
    "best_epoch": ...,
    "best_parameter_path": ...,
    "last_parameter_path": ...,
    "result_root": ...,
}
```

这让后续流程可以明确加载 best checkpoint 做测试。

### 4.3 Tester

`Tester.validate()` 负责：

```text
model.eval()
iterate test_loader
compute per-sample RMSE
optional save visual results
optional log MLflow metrics
optional log MLflow test artifacts
```

当传入：

```python
log_mlflow=True
mlflow_prefix="best_model_test"
mlflow_artifact_path="best_model_test"
```

会记录：

```text
best_model_test_mean_rmse
best_model_test_total_time
best_model_test_avg_time
best_model_test_result_root
best_model_test_dataset_size
```

### 4.4 发布 gate 指标

当前建议 gate 指标：

```text
best_model_test_rmse_for_gate
```

语义：

```text
越低越好
```

使用方式：

```text
current_run.best_model_test_rmse_for_gate
        <
best_finished_previous_run.best_model_test_rmse_for_gate
        |
        v
允许进入后续模型注册/发布步骤
```

当前仓库只记录 gate 指标，尚未实现自动发布脚本。

## 5. DVC 技术实现

### 5.1 DVC 与 Git 的关系

DVC 不把大数据直接提交进 Git，而是提交小的 `.dvc` 指针文件：

```text
Git tracks:
    .dvc/config
    .dvcignore
    <dataset_dir>.dvc
    <data-parent>/.gitignore

DVC cache stores:
    <dataset_dir> actual files
```

关系图：

```text
<dataset_dir>.dvc
        |
        | md5: <data-hash>.dir
        v
local / remote DVC cache
        |
        v
<dataset_dir>/
```

### 5.2 当前数据接管状态

当前仓库已有 NYUv2 对应的 `.dvc` 数据指针，说明该数据目录已经可以被 DVC 描述和追踪。

检查命令：

```bash
dvc status
```

添加或更新数据版本：

```bash
dvc add <dataset_dir>
git add <dataset_dir>.dvc <data-parent>/.gitignore
```

如果数据发生变化，需要重新：

```bash
dvc add <dataset_dir>
git add <dataset_dir>.dvc
```

如果配置了 remote：

```bash
dvc push
```

### 5.3 取消某个数据目录的 DVC 管理

如果要取消某个数据目录的 DVC 跟踪，但保留本地数据，不要使用 `--outs`：

```bash
dvc remove <dataset_dir>.dvc
git add <dataset_dir>.dvc <data-parent>/.gitignore
```

注意：

```text
dvc remove --outs <dataset_dir>.dvc
```

会连输出数据也删除，不适合只取消版本控制的场景。

### 5.4 MLflow 中记录 DVC 元信息

`log_dvc_metadata(args, artifact_dir=None)` 会在训练时自动记录：

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
dvc_status
metadata/dvc_metadata.json
```

对于 DVC 管理的数据目录，理想情况下会记录：

```text
dvc_dataset_path=<dataset_dir>
dvc_dataset_dvc_file=<dataset_dir>.dvc
dvc_dataset_hash=<data-hash>.dir
dvc_dataset_size=<data-size>
dvc_dataset_nfiles=<data-file-count>
```

这样一个 MLflow run 就能同时关联：

```text
code: git_commit
data: dvc_dataset_hash
experiment: mlflow_run_id
```

## 6. Optuna 技术实现

### 6.1 Trial 内部流程

`objective(trial)` 的逻辑：

```text
setup_seed(20 + trial.number)
    |
apply_trial_params(trial)
    |
mlflow.start_run(nested=True)
    |
run_train_and_test(trial)
    |
return mean_rmse
```

每个 trial 会修改全局 `args`：

```python
args.lr
args.batch_size
args.loss
args.step_size
args.lr_gamma
```

这是当前实现简单直接的地方，也意味着同一个 Python 进程内 trial 之间共享 `args` 对象。当前串行 `study.optimize()` 可以接受；如果未来做并行 HPO，需要重构为每个 trial 构造独立 config。

### 6.2 MLflow nested run

结构：

```text
MLflow parent run
    |
    +-- trial_000
    +-- trial_001
    +-- trial_002
```

parent run 记录：

```text
best_trial_number
best_objective_rmse
best_<param>
```

trial run 记录：

```text
trial_number
lr
batch_size
loss
step_size
lr_gamma
objective_rmse
best_model_test_rmse_for_gate
checkpoint artifacts
```

### 6.3 当前限制

- 当前 Optuna storage 使用默认内存存储，进程结束后 study 本身不会持久化。
- 如果需要恢复 study 或分布式调参，建议使用 SQLite/MySQL：

```python
optuna.create_study(
    direction="minimize",
    study_name="d2a2_hpo",
    storage="sqlite:///optuna_d2a2.db",
    load_if_exists=True,
)
```

- 当前 trial 内会创建模型并训练，显存释放依赖 `torch.cuda.empty_cache()`；长时间 HPO 仍建议关注显存碎片。

## 7. PyFunc 服务化实现

### 7.1 为什么需要 PyFunc

D2A2 模型不是单输入模型，而是：

```python
model(guidance, target, mde)
```

直接使用 `mlflow.pytorch.log_model(...); mlflow models serve ...` 时，MLflow 默认请求接口不能自然表达三输入 tensor，因此会出现进入模型后某些输入为 `None` 的问题。

因此新增：

```text
mlflow_transform2pyfunc_model.py
```

将 PyTorch model 包装为：

```python
class D2A2ServingWrapper(mlflow.pyfunc.PythonModel)
```

### 7.2 PyFunc 输入约定

当前 pyfunc wrapper 输入是 pandas DataFrame，每行包含三个 `.pt` 文件路径：

```text
rgb_path
depth_path
mde_path
```

预测时：

```text
torch.load(rgb_path)
torch.load(depth_path)
torch.load(mde_path)
    |
ensure batch dim
    |
model(rgb=rgb, depth=depth, MDE=mde)
    |
return list output
```

测试脚本：

```text
test_d2a2_pyfunc_nyu.py
```

职责：

```text
load NYUv2 sample
save guidance / target / mde to .pt files
optionally write invocation.json
load pyfunc model
predict
print tensor summary
```

示例：

```bash
python test_d2a2_pyfunc_nyu.py \
  --dataset-dir <dataset_dir> \
  --model-uri <best_model_pyfunc_uri> \
  --input-dir /tmp/d2a2_pyfunc_nyu_sample \
  --write-request-json
```

REST 调用：

```bash
mlflow models serve \
  -m <best_model_pyfunc_uri> \
  --env-manager local \
  -h 127.0.0.1 \
  -p 5002 \
  -w 1 \
  -t 180
```

```bash
curl -X POST http://127.0.0.1:5002/invocations \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/d2a2_pyfunc_nyu_sample/invocation.json \
  -o /tmp/d2a2_pyfunc_response.json
```

### 7.3 服务化限制

当前 pyfunc 是可运行验证方案，但仍偏实验性质：

- 输入通过 `.pt` 文件路径传递，不是生产 API 最终形态。
- 未做严格 schema 校验。
- 未封装图像读取、归一化和后处理。
- 生产环境建议用 FastAPI 或更完整的 PyFunc wrapper 接收图片/数组，再统一转换为三输入 tensor。

## 8. 推荐端到端流程

### 8.1 数据准备

```bash
conda activate d2a2

dvc status
dvc pull
```

如果是第一次接管数据：

```bash
dvc add <dataset_dir>
git add <dataset_dir>.dvc <data-parent>/.gitignore .dvc .dvcignore
dvc push
```

### 8.2 单次训练

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir <dataset_dir> \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --cuda_devices 0 \
  --dvc_required \
  --mlflow_run_name d2a2-train-test-v1
```

产物：

```text
MLflow run
    |
    +-- params
    +-- metrics
    +-- checkpoints/
    +-- logs/
    +-- best_model_test/
    +-- metadata/dvc_metadata.json
```

### 8.3 HPO

```bash
python train_test_d2a2_mlflow_optuna_dvc.py \
  --dataset nyu \
  --dataset_dir <dataset_dir> \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --n_trials 1 \
  --cuda_devices 0 \
  --dvc_required \
  --mlflow_run_name d2a2-hpo-v1
```

HPO 正式运行时建议：

```text
固定 git commit
固定 dvc_dataset_hash
增大 n_trials
使用持久化 Optuna storage
保留 parent run id
```

### 8.4 查看 MLflow UI

```bash
mlflow ui \
  --backend-store-uri ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

关注字段：

```text
best_model_test_rmse_for_gate
best_model_test_mean_rmse
best_mean_rmse
best_epoch
dvc_dataset_hash
git_commit
```

## 9. 推荐文件使用策略

### 9.1 推荐作为标准入口

```text
train_test_d2a2_mlflow_dvc.py
train_test_d2a2_mlflow_optuna_dvc.py
mlflow_transform2pyfunc_model.py
test_d2a2_pyfunc_nyu.py
```

### 9.2 可保留作对照

```text
train_d2a2.py
test_d2a2.py
utils.py
train_d2a2_mlflow.py
test_d2a2_mlflow.py
train_test_d2a2_mlflow.py
train_test_d2a2_mlflow_optuna.py
```

### 9.3 可考虑归档或重命名

```text
mlflow_utils_bak.py
```

建议后续把 backup 文件移动到：

```text
archive/
```

或删除前先确认没有入口再 import 它。

## 10. 当前已知风险

### 10.1 参数共享

所有脚本都从 `option.py` 读取全局 `args`。这对单进程串行训练简单有效，但对并行 HPO、服务化配置或单元测试不够理想。

后续建议把配置改造成：

```text
parse_args()
    |
config dataclass / namespace
    |
pass explicitly into builders
```

### 10.2 顶层 import 副作用

部分脚本在 import 阶段就设置：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
```

或构造模型/读取 checkpoint。后续如果要模块化复用，建议把副作用全部移入 `main()`。

### 10.3 DVC 版本偏旧

因为 Python 3.7 约束，当前使用：

```text
dvc==2.10.2
```

如果升级 Python 到 3.8+，可以考虑升级 DVC，但必须重新验证：

```text
dvc
fsspec
mlflow
optuna
torch
transformers
```

### 10.4 DVC remote 尚未标准化

当前已有 DVC 初始化和数据 pointer，但 remote 位置需要团队按实际环境确定。

可选：

```text
local shared path
ssh
s3
oss
nas
```

生产协作中应避免把 credentials 写入 Git。

## 11. 建议后续演进路线

### P0：固化当前标准入口

```text
train_test_d2a2_mlflow_dvc.py
train_test_d2a2_mlflow_optuna_dvc.py
```

补充 Makefile 或 shell script：

```text
make smoke-train
make smoke-hpo
make mlflow-ui
```

### P1：固化数据版本

```text
dvc add <dataset_dir>
dvc remote add ...
dvc push
```

要求每个正式实验记录：

```text
git_commit
dvc_dataset_hash
mlflow_run_id
```

### P2：补充模型发布 gate

实现脚本：

```text
compare_mlflow_runs.py
```

逻辑：

```text
读取当前 run 的 best_model_test_rmse_for_gate
查询历史 FINISHED runs
找最小 RMSE
若当前更低，则允许注册/导出 pyfunc
```

### P3：完善 Optuna 持久化

引入：

```text
sqlite:///optuna_d2a2.db
```

并记录：

```text
study_name
storage_uri
best_trial_number
best_params
```

### P4：构建 DVC pipeline

从单纯 `dvc add` 演进到：

```text
dvc.yaml
    |
    +-- prepare_data
    +-- train
    +-- evaluate
    +-- package_pyfunc
```

目标：

```bash
dvc repro
```

可以复现从数据准备到评估的完整流程。

### P5：服务化工程

生产化服务建议：

```text
FastAPI
    |
receive RGB/depth/MDE files or arrays
    |
preprocess
    |
D2A2 pyfunc / torch model
    |
postprocess
    |
return depth map / save artifact
```

## 12. 总结

当前仓库已经完成了 D2A2 MLOps 化的核心骨架：

```text
DVC       -> 数据版本
Git       -> 代码版本
MLflow    -> 实验追踪与模型产物
Optuna    -> 超参数搜索
PyFunc    -> 三输入模型服务化封装
```

推荐当前标准实验链路：

```text
git pull
    |
dvc pull / dvc checkout
    |
python train_test_d2a2_mlflow_dvc.py
    |
MLflow records params + metrics + checkpoints + DVC metadata
    |
compare best_model_test_rmse_for_gate
    |
optional pyfunc packaging / serving
```

推荐当前标准 HPO 链路：

```text
fixed git_commit + fixed dvc_dataset_hash
    |
python train_test_d2a2_mlflow_optuna_dvc.py
    |
Optuna trials as MLflow nested runs
    |
best params
    |
optional retrain as independent MLflow run
```

这套结构已经可以支持基本的可追溯实验。下一阶段重点不是继续增加入口文件，而是固化标准入口、减少全局副作用、引入持久化 Optuna storage、补齐模型发布 gate，并把数据准备/训练/评估逐步收敛到 DVC pipeline。
