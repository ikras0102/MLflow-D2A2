import functools
import logging
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


total_time = 0.0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_rmse(a, b, min, max):
    a = a * (max - min) + min
    b = b * (max - min) + min
    return np.sqrt(np.mean(np.power(a - b, 2)))


def _sanitize_params(params):
    return {key: "" if value is None else value for key, value in params.items()}


def _mlflow_run_context():
    active_run = mlflow.active_run()
    if active_run is None:
        return mlflow.start_run()
    return nullcontext(active_run)


def _model_to_log(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _create_file_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = os.path.abspath(log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
            return logger

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _as_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_state_dict_flexible(model, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _as_state_dict(checkpoint)

    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())
    model_has_module = any(key.startswith("module.") for key in model_keys)
    checkpoint_has_module = any(key.startswith("module.") for key in checkpoint_keys)

    if checkpoint_has_module and not model_has_module:
        state_dict = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    elif model_has_module and not checkpoint_has_module:
        state_dict = {
            "module.%s" % key if not key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    return model


class MLFlowTrainer(object):
    def __init__(self, args, model, optimizer, scheduler, criterion, train_loader, test_loader):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.maxepoch = args.epoch
        self.nowepoch = 0
        self.upscale_func = functools.partial(
            F.interpolate, mode="bicubic", align_corners=False
        )

        s = datetime.now().strftime("%Y%m%d%H%M%S")
        result_root = "%s/%s-x%s-%s" % (args.trainresult, args.model_file, args.scale, s)
        os.makedirs(result_root, exist_ok=True)
        self.result_root = result_root
        self.best_parameter_path = os.path.join(self.result_root, "best_parameter")
        self.last_parameter_path = os.path.join(self.result_root, "last_parameter")
        self.logger = _create_file_logger(
            "d2a2.train.%s" % s, os.path.join(result_root, "train.log")
        )
        self.logger.info("args:\n%s\n" % self.args)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rmse = np.zeros(self.test_loader.__len__())
        t = tqdm(iter(self.test_loader), leave=True, total=self.test_loader.__len__())
        for idx, data in enumerate(t):
            guidance = data["guidance"].cuda()
            target = data["target"].cuda()
            gt = data["gt"].cuda()
            min_value = data["min"]
            max_value = data["max"]
            mde = data["mde"].cuda()
            out = self.model(guidance, target, mde)
            gt_ = gt[0, 0].cpu().numpy()
            out_ = out[0, 0].cpu().numpy()
            if self.args.dataset == "nyu":
                gt_ = gt_[6:-6, 6:-6]
                out_ = out_[6:-6, 6:-6]
            rmse[idx] = calc_rmse(gt_, out_, min_value.numpy(), max_value.numpy())
            t.set_description("[validate] rmse: %f" % rmse[:idx + 1].mean())
            t.refresh()
        return rmse

    def train(self):
        with _mlflow_run_context():
            model_to_log = _model_to_log(self.model)
            mlflow.log_params(_sanitize_params(vars(self.args)))
            mlflow.log_params(
                _sanitize_params(
                    {
                        "optimizer": self.optimizer.__class__.__name__,
                        "scheduler": self.scheduler.__class__.__name__,
                        "criterion": self.criterion.__class__.__name__,
                        "model_class": model_to_log.__class__.__name__,
                        "train_dataset_size": len(self.train_loader.dataset),
                        "test_dataset_size": len(self.test_loader.dataset),
                        "train_result_root": self.result_root,
                    }
                )
            )

            max_epoch = self.maxepoch
            validate_interval = max(1, getattr(self.args, "validate_interval", 5))
            best_rmse = float("inf")
            best_epoch = 0

            for epoch in range(max_epoch):
                self.nowepoch = epoch + 1
                self.model.train()
                running_loss = 0.0
                epoch_loss = 0.0
                t = tqdm(iter(self.train_loader), leave=True, total=self.train_loader.__len__())
                for idx, data in enumerate(t):
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    guidance = data["guidance"].cuda()
                    target = data["target"].cuda()
                    gt = data["gt"].cuda()
                    mde = data["mde"].cuda()
                    out = self.model(guidance, target, mde)
                    loss = self.criterion(out, gt)
                    loss.backward()
                    self.optimizer.step()
                    batch_loss = loss.data.item()
                    running_loss += batch_loss
                    epoch_loss += batch_loss
                    if idx % 50 == 0:
                        running_loss /= 50
                        t.set_description(
                            "[train epoch(L1):%d] loss: %.10f"
                            % (self.nowepoch, running_loss)
                        )
                        t.refresh()
                        self.logger.info(
                            "epoch:%d running_loss:%.10f"
                            % (self.nowepoch, running_loss)
                        )
                        mlflow.log_metrics(
                            {
                                "train_running_loss": float(running_loss),
                                "train_batch_loss": float(batch_loss),
                            },
                            step=epoch * self.train_loader.__len__() + idx,
                        )

                epoch_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                self.logger.info(
                    "epoch:%d optimizer_lr:%s" % (self.nowepoch, epoch_lr)
                )
                mlflow.log_metrics(
                    {
                        "train_epoch_loss": float(epoch_loss / self.train_loader.__len__()),
                        "optimizer_lr": float(epoch_lr),
                    },
                    step=self.nowepoch,
                )
                torch.save(self.model.state_dict(), self.last_parameter_path)

                if epoch % validate_interval == 0:
                    rmse = self.validate()
                    mean_rmse = float(rmse.mean())
                    self.logger.info(
                        "epoch:%d --------mean_rmse:%.10f "
                        % (self.nowepoch, mean_rmse)
                    )
                    mlflow.log_metrics({"val_mean_rmse": mean_rmse}, step=self.nowepoch)
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_epoch = self.nowepoch
                        torch.save(self.model.state_dict(), self.best_parameter_path)
                        mlflow.log_metrics(
                            {
                                "best_mean_rmse": float(best_rmse),
                                "best_epoch": float(best_epoch),
                            },
                            step=self.nowepoch,
                        )
                        mlflow.log_artifact(
                            self.best_parameter_path, artifact_path="checkpoints"
                        )
                    self.logger.info(
                        "best_epoch:%d --------best_mean_rmse:%.10f "
                        % (best_epoch, best_rmse)
                    )

                if epoch >= max_epoch - 5:
                    torch.save(
                        self.model.state_dict(),
                        "%s/%s_parameter" % (self.result_root, epoch),
                    )

            mlflow.pytorch.log_model(_model_to_log(self.model), "model")
            if os.path.exists(self.best_parameter_path):
                load_state_dict_flexible(self.model, self.best_parameter_path)
                mlflow.pytorch.log_model(_model_to_log(self.model), "best_model")
                self.logger.info("loaded best_parameter for downstream test")
            if os.path.exists(self.last_parameter_path):
                mlflow.log_artifact(self.last_parameter_path, artifact_path="checkpoints")
            mlflow.log_artifact(
                os.path.join(self.result_root, "train.log"), artifact_path="logs"
            )

            return {
                "best_rmse": None if best_epoch == 0 else float(best_rmse),
                "best_epoch": best_epoch,
                "best_parameter_path": self.best_parameter_path
                if os.path.exists(self.best_parameter_path)
                else None,
                "last_parameter_path": self.last_parameter_path
                if os.path.exists(self.last_parameter_path)
                else None,
                "result_root": self.result_root,
            }


class Tester(object):
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader

        s = datetime.now().strftime("%Y%m%d%H%M%S")
        result_root = "%s/%s-%s-x%s-%s" % (
            args.testresult,
            args.model_file,
            args.dataset,
            args.scale,
            s,
        )
        os.makedirs(result_root, exist_ok=True)
        if self.args.save:
            save_depth_root = result_root + "/depthsr"
            save_hotmap_root = result_root + "/hotmap"
            os.makedirs(save_depth_root, exist_ok=True)
            os.makedirs(save_hotmap_root, exist_ok=True)
            self.save_depth_root = save_depth_root
            self.save_hotmap_root = save_hotmap_root
        self.result_root = result_root
        self.logger = _create_file_logger(
            "d2a2.test.%s" % s, os.path.join(result_root, "test.log")
        )

    @torch.no_grad()
    def validate(
        self,
        mlflow_prefix=None,
        mlflow_step=None,
        log_mlflow=False,
        mlflow_artifact_path=None,
    ):
        self.model.eval()
        rmse = np.zeros(self.test_loader.__len__())
        global total_time
        total_time = 0.0
        t = tqdm(iter(self.test_loader), leave=True, total=self.test_loader.__len__())
        for idx, data in enumerate(t):
            guidance = data["guidance"].cuda()
            target = data["target"].cuda()
            gt = data["gt"].cuda()
            min_value = data["min"]
            max_value = data["max"]
            mde = data["mde"].cuda()
            begin_time = time.time()
            out = self.model(guidance, target, mde)
            end_time = time.time()
            total_time = total_time + (end_time - begin_time)
            errormap = torch.abs(gt - out)
            gt_ = gt[0, 0].cpu().numpy()
            out_ = out[0, 0].cpu().numpy()
            if self.args.dataset == "nyu":
                gt_ = gt_[6:-6, 6:-6]
                out_ = out_[6:-6, 6:-6]
            rmse[idx] = calc_rmse(gt_, out_, min_value.numpy(), max_value.numpy())
            if self.args.save:
                out_depth = (
                    out[0][0].cpu().numpy() * (max_value.numpy() - min_value.numpy())
                    + min_value.numpy()
                )
                gt_depth = (
                    gt[0][0].cpu().numpy() * (max_value.numpy() - min_value.numpy())
                    + min_value.numpy()
                )
                lr_depth = (
                    target[0][0].cpu().numpy()
                    * (max_value.numpy() - min_value.numpy())
                    + min_value.numpy()
                )
                errormap = (
                    errormap[0][0].cpu().numpy()
                    * (max_value.numpy() - min_value.numpy())
                    + min_value.numpy()
                )
                plt.imsave(
                    os.path.join(self.save_hotmap_root, "%s_sr_color.png" % idx),
                    out_depth,
                    cmap="plasma",
                )
                plt.imsave(
                    os.path.join(self.save_hotmap_root, "%s_errormap.png" % idx),
                    errormap,
                    cmap="afmhot",
                )
                plt.imsave(
                    os.path.join(self.save_hotmap_root, "%s_gt_color.png" % idx),
                    gt_depth,
                    cmap="plasma",
                )
                plt.imsave(
                    os.path.join(self.save_hotmap_root, "%s_lr_color.png" % idx),
                    lr_depth,
                    cmap="plasma",
                )
            t.set_description("[validate] rmse: %f" % rmse[:idx + 1].mean())
            t.refresh()
            self.logger.info("idx:%d rmse:%.10f" % (idx, rmse[idx]))

        mean_rmse = float(rmse.mean())
        avg_time = float(total_time / max(1, self.test_loader.__len__()))
        self.logger.info("mean rmse:%.10f\n\n\n" % mean_rmse)
        self.logger.info("total_time:%.10f" % total_time)

        if log_mlflow and mlflow.active_run() is not None:
            prefix = mlflow_prefix or "test"
            mlflow.log_metrics(
                {
                    "%s_mean_rmse" % prefix: mean_rmse,
                    "%s_total_time" % prefix: float(total_time),
                    "%s_avg_time" % prefix: avg_time,
                },
                step=mlflow_step,
            )
            mlflow.log_params(
                _sanitize_params(
                    {
                        "%s_result_root" % prefix: self.result_root,
                        "%s_dataset_size" % prefix: len(self.test_loader.dataset),
                    }
                )
            )
            artifact_path = mlflow_artifact_path or prefix
            mlflow.log_artifacts(self.result_root, artifact_path=artifact_path)

        return rmse
