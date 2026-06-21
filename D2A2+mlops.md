# D2A2 + MLOps 技术报告

## 1. 背景

D2A2 原始代码主要面向单次训练和测试，核心流程是：

```text
NYUv2 / Lu / Middlebury / RGBDD
        |
        v
Dataset + DataLoader
        |
        v
D2A2(guidance, target, mde)
        |
        v
train / validate / test
        |
        v
result/trainresult, result/testresult
```

当前仓库在此基础上引入了三类 MLOps 能力：

- MLflow：记录训练参数、指标、checkpoint、测试结果和模型 artifact。
- Optuna：围绕 RMSE 指标做超参数搜索。
- DVC：管理数据目录版本，使实验结果可以追溯到具体数据快照。

三者的职责边界如下：

```text
             +------------------+
             |      Git         |
             | code/config/dvc  |
             +---------+--------+
                       |
                       v
+----------+    +------+-------+    +----------------+
|   DVC    | -> | Training     | -> |    MLflow      |
| data ver |    | D2A2 + torch |    | params/metrics |
+----------+    +------+-------+    | artifacts      |
                       ^            +----------------+
                       |
                 +-----+------+
                 |  Optuna    |
                 | HPO trials |
                 +------------+
```

## 2. 代码入口

### 2.1 MLflow 单次训练 + 测试

入口文件：

```text
train_test_d2a2_mlflow.py
```

主要流程：

```text
build_model
    |
build_train_loader + build_test_loader
    |
build_optimizer_scheduler + build_criterion
    |
MLFlowTrainer.train()
    |
save best_parameter / last_parameter
    |
load best_parameter
    |
Tester.validate()
    |
log best_model_test_rmse_for_gate
```

核心 MLflow 记录项：

- params：`scale`、`dataset`、`dataset_dir`、`lr`、`batch_size`、`loss`、`step_size`、`lr_gamma` 等。
- metrics：`train_epoch_loss`、`val_mean_rmse`、`best_mean_rmse`、`best_model_test_mean_rmse`、`best_model_test_rmse_for_gate`。
- artifacts：best checkpoint、last checkpoint、训练日志、测试输出。
- tags：`pipeline`、`run_id`、`git_commit`、`dvc_status`。

示例命令：

```bash
python train_test_d2a2_mlflow.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2 \
  --scale 16 \
  --epoch 1 \
  --batch_size 1 \
  --validate_interval 1 \
  --cuda_devices 0 \
  --mlflow_run_name d2a2-mlflow-smoke
```

### 2.2 Optuna 超参数搜索

入口文件：

```text
train_test_d2a2_mlflow_optuna.py
```

Optuna 以 trial 为单位修改训练参数，并把每次 trial 作为 MLflow nested run 记录。

```text
parent MLflow run: optuna_hpo
        |
        +-- trial_000 -> train -> test -> objective_rmse
        |
        +-- trial_001 -> train -> test -> objective_rmse
        |
        +-- trial_002 -> train -> test -> objective_rmse
        |
        v
best_trial + best_params + best_objective_rmse
```

当前搜索空间：

```text
lr         log uniform: 1e-5 ~ 1e-3
batch_size categorical: 2, 4, 8
loss       categorical: l1, mask
step_size  categorical: 10, 20, 30
lr_gamma   categorical: 0.3, 0.5, 0.7
```

示例命令：

```bash
python train_test_d2a2_mlflow_optuna.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2 \
  --scale 16 \
  --epoch 1 \
  --validate_interval 1 \
  --n_trials 1 \
  --cuda_devices 0 \
  --mlflow_run_name d2a2-optuna-smoke
```

## 3. DVC 数据版本控制

### 3.1 当前目标

DVC 的目标不是替代 MLflow，而是补上“数据版本”这一层：

```text
Git commit  ->  code version
DVC .dvc    ->  data version
MLflow run  ->  experiment version
```

实验可复现时至少需要三元组：

```text
(git_commit, dvc_dataset_hash, mlflow_run_id)
```

### 3.2 数据目录

当前建议使用：

```text
datasets/NYUv2
```

而不是根目录下的：

```text
NYUv2
```

原因是实际检查中 `datasets/NYUv2` 是当前训练/pyfunc 测试可用的数据目录，根目录 `NYUv2` 曾出现过 `Depth/*.npy` 读取异常。

### 3.3 DVC 初始化与数据接管

初始化：

```bash
dvc init
git add .dvc .dvcignore
```

接管 NYUv2：

```bash
dvc add datasets/NYUv2
git add datasets/NYUv2.dvc datasets/.gitignore .gitignore
```

如果要推送到远程缓存：

```bash
dvc remote add -d localstore /path/to/dvc-store
dvc push
git add .dvc/config
```

恢复数据：

```bash
git pull
dvc pull
```

复现旧版本：

```bash
git checkout <commit>
dvc checkout
```

### 3.4 DVC 版训练入口

新增 DVC sketch 文件：

```text
train_test_d2a2_mlflow_dvc.py
train_test_d2a2_mlflow_optuna_dvc.py
dvc_training_utils.py
```

DVC 版单次训练：

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2 \
  --epoch 1 \
  --validate_interval 1 \
  --batch_size 1 \
  --dvc_required
```

DVC 版 Optuna：

```bash
python train_test_d2a2_mlflow_optuna_dvc.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2 \
  --epoch 1 \
  --validate_interval 1 \
  --batch_size 1 \
  --n_trials 1 \
  --dvc_required
```

如需训练前自动拉取数据：

```bash
--dvc_pull
```

如需指定 DVC remote：

```bash
--dvc_remote <remote-name>
```

## 4. MLflow 与 DVC 元信息关联

`mlflow_utils.py` 中新增了 `log_dvc_metadata(args, artifact_dir=None)`，训练时自动尝试记录：

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
```

记录逻辑是 best effort：

- DVC 未安装时，训练不直接失败，除非显式传入 `--dvc_required`。
- DVC 已初始化且存在 `.dvc` 文件时，会解析 `.dvc` sidecar 中的数据 hash、size、nfiles。
- 同时会把完整 JSON 元信息写入 MLflow artifact：`metadata/dvc_metadata.json`。

链路如下：

```text
datasets/NYUv2.dvc
        |
        | parse outs[0].md5 / size / nfiles
        v
log_dvc_metadata()
        |
        +--> mlflow.log_params(...)
        |
        +--> mlflow.set_tag("dvc_status", ...)
        |
        +--> metadata/dvc_metadata.json
```

## 5. 模型产物与服务化

### 5.1 PyTorch 模型 artifact

训练阶段可以记录：

```text
model
best_model
checkpoints/best_parameter
checkpoints/last_parameter
```

其中 `best_model_test_rmse_for_gate` 是当前建议用于模型发布 gate 的主指标。

```text
lower RMSE is better

current_run.best_model_test_rmse_for_gate
        <
best_previous_run.best_model_test_rmse_for_gate
        |
        v
allow publish
```

### 5.2 PyFunc 封装

D2A2 的 forward 接口是三输入：

```python
model(guidance, target, mde)
```

MLflow `mlflow.pytorch.log_model` 默认服务接口更适合单 tensor 或 dataframe 输入，不能直接表达 D2A2 的三输入语义。因此引入了：

```text
mlflow_transform2pyfunc_model.py
test_d2a2_pyfunc_nyu.py
```

服务化链路：

```text
best_model artifact
        |
        v
mlflow_transform2pyfunc_model.py
        |
        v
best_model_pyfunc
        |
        v
mlflow models serve
        |
        v
JSON request: guidance + target_lr_depth + mde
        |
        v
prediction: HR depth
```

本地 pyfunc 测试形式：

```bash
python test_d2a2_pyfunc_nyu.py \
  --dataset-dir datasets/NYUv2 \
  --model-uri <best_model_pyfunc_uri> \
  --input-dir /tmp/d2a2_pyfunc_nyu_sample \
  --write-request-json
```

REST 测试形式：

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

## 6. 推荐端到端流程

### 6.1 单次实验

```text
git pull
    |
dvc pull / dvc checkout
    |
python train_test_d2a2_mlflow_dvc.py
    |
MLflow run:
  - params
  - metrics
  - checkpoints
  - dvc metadata
    |
compare best_model_test_rmse_for_gate
```

### 6.2 HPO 实验

```text
git + dvc fixed
    |
parent MLflow run
    |
Optuna study
    |
trial runs
    |
best params
    |
optional retrain with best params
```

推荐实践：

- HPO 阶段固定 `git_commit` 和 `dvc_dataset_hash`。
- trial 内只改变超参数，不改变数据。
- 用 validation RMSE 选择 trial，用独立 test RMSE 做最终报告。
- 最终模型最好用 best params 再做一次独立 retrain，并生成单独 MLflow run。

## 7. 环境兼容性

当前环境是 Python 3.7，因此 DVC 不能直接使用最新版。当前兼容组合为：

```text
python==3.7.16
dvc==2.10.2
fsspec==2022.2.0
mlflow==1.30.0
optuna==3.0.0
torch==1.8.0
torchvision==0.9.0
```

注意点：

- `dvc==2.10.2` 与 `fsspec==2023.1.0` 不兼容，会触发 `fsspec_loop` import error。
- 已将 `environment.yaml` 中的 `fsspec` pin 到 `2022.2.0`。
- `pip check` 当前未发现 broken requirements。
- 如果后续升级 Python 到 3.8+，可以考虑升级 DVC 到更高版本，但要重新验证 MLflow、Optuna、Torch、Transformers 的兼容性。

## 8. 当前限制

### 8.1 DVC

- 目前 DVC 管理的是数据目录指针和缓存，尚未定义完整 `dvc.yaml` pipeline。
- 如果只使用 `dvc add datasets/NYUv2`，DVC 记录的是数据快照，不记录预处理步骤。
- 如后续有数据生成脚本，建议把预处理加入 `dvc stage add`。

### 8.2 MLflow

- 当前主要使用本地 `mlruns/` file store。
- 尚未接入 Model Registry。
- 模型发布 gate 目前只形成指标字段，尚未实现自动发布脚本。

### 8.3 Optuna

- 当前搜索空间偏小，适合先跑通流程。
- Study 默认使用内存存储，长时间实验建议使用 SQLite/MySQL storage。
- 多 GPU / 多进程 HPO 还需要额外处理资源隔离。

### 8.4 服务化

- 原生 PyTorch MLflow serving 不适合直接服务 D2A2 三输入模型。
- PyFunc 已提供可行封装方向，但生产服务还需要补齐输入校验、批处理、异常处理和输出格式规范。

## 9. 后续技术路线

优先级建议：

```text
P0: 固化数据版本
    - dvc add datasets/NYUv2
    - 配置 remote
    - 在 MLflow 中稳定记录 dvc_dataset_hash

P1: 固化实验入口
    - 保留 train_test_d2a2_mlflow_dvc.py 作为标准训练入口
    - 保留 train_test_d2a2_mlflow_optuna_dvc.py 作为标准 HPO 入口
    - 将烟测参数写入 README 或 Makefile

P2: 引入模型发布 gate
    - 查询历史 run
    - 比较 best_model_test_rmse_for_gate
    - 通过后注册或导出 best_model_pyfunc

P3: 完善 DVC pipeline
    - data prepare
    - train
    - evaluate
    - package pyfunc

P4: 服务化工程
    - FastAPI / MLflow pyfunc endpoint
    - 输入文件解析
    - GPU 推理
    - 输出深度图和指标
```

最终目标链路：

```text
        Git commit
            |
            v
        DVC data hash
            |
            v
      MLflow experiment
            |
            v
      Optuna best params
            |
            v
      Best checkpoint / pyfunc
            |
            v
      Gate by test RMSE
            |
            v
      Deploy / archive / reproduce
```

