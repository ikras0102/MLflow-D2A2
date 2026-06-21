# MLflow + D2A2 v1.0

## 目标

本次改动把 D2A2 的训练和 best model 测试串到同一个 MLflow run 中：

1. 训练阶段按 `validate_interval` 做验证，记录 `val_mean_rmse`。
2. 当前验证 RMSE 优于本次训练历史最优时，保存 `best_parameter`，并记录 `best_mean_rmse`、`best_epoch`。
3. 训练结束后加载 `best_parameter`，对 best model 再跑一次 test。
4. test 指标写回同一个 MLflow run，后续可以用 `best_model_test_rmse_for_gate` 做“优于历史最优才发布”的控制条件。

`mlruns/1/<hash>/` 里的 `<hash>` 就是 MLflow 的 `run_id`。代码里不需要遍历目录拿它，在 `mlflow.start_run()` 里可以直接通过 `run.info.run_id` 获取。

## 修改文件

### `train_test_d2a2_mlflow.py`

新增统一入口，执行完整流程：

```text
build model -> build train/test dataloader -> train -> load best_parameter -> test best model -> log metrics/artifacts
```

关键记录项：

- MLflow tag: `pipeline=train_then_test_best_model`
- MLflow tag: `run_id=<当前 run id>`
- MLflow metric: `best_mean_rmse`
- MLflow metric: `best_epoch`
- MLflow metric: `best_model_test_mean_rmse`
- MLflow metric: `best_model_test_total_time`
- MLflow metric: `best_model_test_avg_time`
- MLflow metric: `best_model_test_rmse_for_gate`
- MLflow param: `tested_checkpoint_path`

其中 `best_model_test_rmse_for_gate` 是后续发布控制建议使用的主指标，RMSE 越低越好。

### `mlflow_utils.py`

主要调整：

- `MLFlowTrainer.train()` 支持复用外层已存在的 MLflow run。这样 train 和 test 可以写入同一个 run。
- 训练过程保存并记录 `checkpoints/best_parameter`、`checkpoints/last_parameter`。
- 训练结束后把 final model 记录到 `model`，把 best checkpoint 加载后记录到 `best_model`。
- `Tester.validate()` 增加 MLflow 记录参数，可把 test RMSE、耗时和 test 输出目录作为同一 run 的指标/产物。
- 增加 `load_state_dict_flexible()`，兼容 `DataParallel` 产生的 `module.` 前缀。
- 训练日志和测试日志改为各自独立 logger，避免同一个进程内 `logging.basicConfig()` 只生效一次导致日志串写。

### `option.py`

新增参数：

```bash
--validate_interval       验证和更新 best checkpoint 的 epoch 间隔，默认 5
--mlflow_experiment       MLflow experiment 名称，默认 MLflow+D2A2
--mlflow_run_name         可选 run name
--cuda_devices            CUDA_VISIBLE_DEVICES，默认 0,1
```

同时把 `--lr` 默认值从字符串修正为浮点数 `0.0001`。

## Demo

### 1. 快速跑通 1 个 epoch

```bash
python train_test_d2a2_mlflow.py \
  --dataset nyu \
  --dataset_dir datasets/NYUv2/ \
  --scale 16 \
  --epoch 1 \
  --batch_size 4 \
  --validate_interval 1 \
  --cuda_devices 0 \
  --mlflow_run_name d2a2-train-test-demo
```

运行结束后终端会打印：

```text
MLflow run_id: <run_id>
best checkpoint: <path/to/best_parameter>
best_model_test_mean_rmse: <rmse>
```

### 2. 启动 MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

打开页面后选择 `MLflow+D2A2` experiment，可以看到该 run 下同时包含训练指标和 best model 的测试指标。

### 3. 查看后续 gate 需要的指标

后续联调“优于历史最优才发布”时，建议以同一个 experiment 下已完成 run 的 `best_model_test_rmse_for_gate` 为比较对象：

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("MLflow+D2A2")
current_run_id = "<current-run-id>"
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        "attributes.status = 'FINISHED' "
        "and metrics.best_model_test_rmse_for_gate >= 0"
    ),
    order_by=["metrics.best_model_test_rmse_for_gate ASC"],
)

current_rmse = 10.0
previous_runs = [run for run in runs if run.info.run_id != current_run_id]
best_previous_rmse = (
    previous_runs[0].data.metrics["best_model_test_rmse_for_gate"]
    if previous_runs
    else float("inf")
)
can_publish = current_rmse < best_previous_rmse
```

这部分当前没有接模型发布或注册，只保留指标和比较入口，方便后续和 registry 或发布脚本联调。

### 4. 模拟模型部署

当前训练脚本已经把最终模型记录到 `model`，把验证集最优 checkpoint 对应的模型记录到 `best_model`。模拟部署时建议优先部署 `best_model`：

```bash
conda activate d2a2

export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
export CUDA_VISIBLE_DEVICES=0

RUN_ID=4a682a668e7643e7978540d626d7433e

mlflow models serve \
  --model-uri "runs:/${RUN_ID}/best_model" \
  --host 0.0.0.0 \
  --port 5001 \
  --workers 1 \
  --env-manager local
```

如果不想依赖 `runs:/` 解析，也可以直接用本地 artifact 路径：

```bash
conda activate d2a2

export CUDA_VISIBLE_DEVICES=0

RUN_ID=4a682a668e7643e7978540d626d7433e

mlflow models serve \
  --model-uri "./mlruns/1/${RUN_ID}/artifacts/best_model" \
  --host 0.0.0.0 \
  --port 5001 \
  --workers 1 \
  --env-manager local
```

服务启动后可以先做健康检查：

```bash
curl http://127.0.0.1:5001/ping
```

注意：这一步的目标是模拟“模型 artifact 可以被 MLflow 拉起成服务”。D2A2 的 `forward(rgb, depth, MDE)` 是三输入张量接口，MLflow 默认 pyfunc REST 接口更适合单个 dataframe/tensor 输入；真实业务推理时建议再封装一层自定义 `pyfunc` 或 FastAPI 服务，负责图片/深度图读取、归一化、`rgb/depth/MDE` 组装、后处理和结果保存。

上面这个起法可以正常启动一个服务,但是如果我们尝试:

```bash
  curl -i -X POST http://127.0.0.1:5001/invocations \
    -H 'Content-Type: application/json' \
    -d '{"columns":["x"],"data":[[0.0]]}'
```

这个请求已经进入了模型的 forward()，报错停在：

```PlainText
  D2A2_depthanything_L_scale16.py", line 546, in forward
  ref=torch.cat([ref, MDE], dim=1)
  TypeError: expected Tensor as element 1 in argument 0, but got NoneType
```
结论是：服务能启动，也能接收 /invocations 并进入模型；但当前用 mlflow.pytorch 直接 serve 的接口只能传一个 tensor，而D2A2 模型需要三个输入：

model(guidance, target, mde)

所以它现在还不能完成一次真正的三输入推理。要完整交互，需要把模型重新封装成 mlflow.pyfunc.PythonModel，在 predict() 里把请求里的guidance / target / mde 转成三个 tensor 后再调用模型。

这也是后续的一个**修复方向**。

修复：引入脚本`mlflow_transform2pyfunc_model.py`

这个脚本load一个模型然后包装成pyfunc，运行后会输出一个新的RUN_ID，如1d0b79e73e70467f8e02fd6a1a98ff6d

• 我测通了。关键点是：NYU 要用 datasets/NYUv2，不是根目录下的 NYUv2。我抽查发现 NYUv2/Depth/*.npy 是坏的，np.load 会报 cannot reshape
  array...；而 datasets/NYUv2 正常。

  我加了一个复用脚本：test_d2a2_pyfunc_nyu.py，它会从 NYU 样本生成 pyfunc 需要的三个 .pt 输入，然后直接
  mlflow.pyfunc.load_model(...).predict(...)。

  本地 pyfunc 测试命令：

  /root/miniconda3/envs/d2a2/bin/python test_d2a2_pyfunc_nyu.py \
    --dataset-dir datasets/NYUv2 \
    --model-uri mlruns/3/1d0b79e73e70467f8e02fd6a1a98ff6d/artifacts/best_model_pyfunc \
    --input-dir /tmp/d2a2_pyfunc_nyu_sample \
    --write-request-json

  我这里跑出来成功：

  guidance: shape=(3, 480, 640)
  target_lr_depth: shape=(1, 30, 40)
  mde: shape=(1, 480, 640)
  prediction: shape=(1, 1, 480, 640), min=-0.040676, max=0.983403, mean=0.580838

  REST 也测通了。启动服务：

  MLFLOW_TRACKING_URI=file:///root/autodl-tmp/.autodl/D2A2-master1/mlruns \
  /root/miniconda3/envs/d2a2/bin/mlflow models serve \
    -m /root/autodl-tmp/.autodl/D2A2-master1/mlruns/3/1d0b79e73e70467f8e02fd6a1a98ff6d/artifacts/best_model_pyfunc \
    --env-manager local \
    -h 127.0.0.1 -p 5002 -w 1 -t 180

  请求：

  curl -X POST http://127.0.0.1:5002/invocations \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/d2a2_pyfunc_nyu_sample/invocation.json \
    -o /tmp/d2a2_pyfunc_response.json

  结果是 HTTP_STATUS=200，响应解析后维度也是 (1, 1, 480, 640)。我已经把测试用的 MLflow 服务停掉了。

## 后续 MLOps 实践方向

### 1. 数据版本控制

建议优先做。当前训练结果已经能被 MLflow 记录，下一步需要保证“同一个模型结果能追溯到同一份数据”。

可落地点：

- 使用 DVC 或类似工具管理 `datasets/NYUv2/` 的原始数据、预处理数据和 train/test split。
- 在 MLflow run 中记录 `data_version`、`split_version`、`dataset_name`、`dataset_dir`、`scale`、样本数量等信息。
- 固定验证集和测试集，不把 test set 用作调参反馈。
- 如果数据不能直接进版本库，至少记录数据清单文件，例如文件路径、大小、hash、生成时间。

### 2. 训练参数调优

可以作为第二阶段做。当前 `best_model_test_rmse_for_gate` 已经能作为主目标指标，后续可以用 Optuna、Ray Tune 或简单 grid/random search 做参数搜索。

建议优先搜索：

- `lr`
- `batch_size`
- `loss`
- `input_size`
- `validate_interval`
- `step_size`
- `n_resblocks`
- `n_feats`
- `augment`

实践建议：

- 每组参数作为一个 MLflow run，调参任务作为 parent run，具体训练作为 nested run。
- 调参只看 validation 指标，最终只用 test 指标做一次确认。
- 除了 RMSE，也记录训练耗时、单样本推理耗时、显存占用，避免只追求精度导致部署不可用。

### 3. 基于 test 结果决定是否更新模型

这是当前代码最接近可以落地的一步。已有指标 `best_model_test_rmse_for_gate`，后续可以做一个独立的 gate 脚本：

```text
读取当前 run 的 best_model_test_rmse_for_gate
-> 查找同 experiment、同 dataset、同 scale 下历史已发布模型
-> 对比当前 Production 模型的 test RMSE
-> 当前模型明显更好才注册或更新部署目标
```

建议 gate 条件不要只看 RMSE：

- 当前 run 必须是 `FINISHED`。
- `dataset`、`scale`、`split_version` 必须和 Production 模型一致。
- `best_model_test_rmse_for_gate` 至少优于 Production 模型一个最小阈值，例如 `current_rmse < production_rmse - min_delta`。
- 推理耗时不能明显变差，例如 `best_model_test_avg_time` 不超过设定阈值。
- 关键 artifact 必须存在：`best_model`、`checkpoints/best_parameter`、测试日志和测试结果目录。

通过 gate 后可以进入 MLflow Model Registry：

```text
runs:/<run_id>/best_model
-> register model
-> 转到 Staging
-> 人工确认或自动晋升为 Production
-> 部署服务读取 Production 对应的 model uri
```

### 4. 模型注册、发布和回滚

在部署模拟跑通后，可以把“服务加载哪个模型”从手写 run_id 改成 Model Registry 的 stage 驱动。当前环境是 MLflow 1.30.0，建议先使用 `Staging` / `Production` / `Archived` 这类 stage；如果后续升级到 MLflow 2.x，再考虑使用 alias。

建议流程：

- `Staging`: 新训练完成并通过基础测试，等待确认。
- `Production`: 当前线上或默认部署模型。
- `Archived`: 被替换但保留可回滚的模型。

这样部署命令可以逐步从：

```text
runs:/<run_id>/best_model
```

演进为：

```text
models:/D2A2/Production
```

### 5. 自动触发训练

这个可以放到较后阶段。前面的数据版本、参数记录、gate、registry 没有稳定前，自动触发训练容易产生很多不可追溯的 run。

适合的触发条件：

- 新数据版本进入 DVC remote。
- 训练代码或模型结构更新。
- 定时任务，例如每周或每月重训。
- 线上监控发现数据分布漂移或性能下降。

可选实现：

- GitHub Actions / GitLab CI
- Airflow / Prefect
- cron + shell 脚本
- 平台化训练任务，例如 Kubernetes Job

### 6. 推理服务封装和监控

MLflow `models serve` 适合做本地模拟和验收 artifact，但 D2A2 真实推理建议单独封装服务层。

服务层建议包含：

- 输入校验：图片尺寸、通道数、depth 范围、缺失字段。
- 预处理：读取 RGB、低分辨率 depth、MDE，执行和训练一致的归一化。
- 推理：加载 MLflow Production 模型或 checkpoint。
- 后处理：反归一化、保存深度图、返回结果路径或数组。
- 监控：请求数量、失败率、平均延迟、P95 延迟、GPU 显存、输入分布。

### 7. 基础 CI 和可复现实验

为了避免后续 MLOps 自动化后问题变得难排查，建议补一组轻量检查：

- `train_test_d2a2_mlflow.py --epoch 1 --validate_interval 1` 的 smoke test。
- `load_state_dict_flexible()` 的 DataParallel/non-DataParallel checkpoint 加载测试。
- dataloader 输出字段检查：`guidance`、`target`、`gt`、`mde`、`min`、`max`。
- MLflow run 必须包含关键指标：`best_mean_rmse`、`best_epoch`、`best_model_test_rmse_for_gate`。
- 固定随机种子、记录环境版本、记录 CUDA 和 PyTorch 版本。

建议优先级：

```text
数据版本控制
-> gate 脚本和模型注册
-> 推理服务封装
-> 参数调优
-> CI smoke test
-> 自动触发训练
-> 线上监控和周期重训
```
