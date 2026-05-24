import os

from option import args

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import functools

import mlflow
import torch
import torch.nn.functional as F

from datasets import *
from mlflow_utils import (
    MLFlowTrainer,
    Tester,
    load_state_dict_flexible,
    setup_seed,
)


if args.scale == 4:
    from models.D2A2_depthanything import D2A2
elif args.scale == 8:
    from models.D2A2_depthanything_L_scale8 import D2A2
elif args.scale == 16:
    from models.D2A2_depthanything_L_scale16 import D2A2
else:
    raise NotImplementedError("Scale %s not found" % args.scale)


def mask_normal(tensor, mean=10):
    mask = tensor < tensor.mean() / mean
    tensor[mask] = (tensor[mask] - tensor[mask].min()) / (
        tensor[mask].max() - tensor[mask].min()
    )
    tensor[~mask] = 1
    return tensor


class MaskLoss(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.alpha = torch.nn.Parameter(torch.tensor(1.5))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        self.weig_func2 = lambda x, y, z: torch.exp(x * y) * z
        self.upscale_func = functools.partial(
            F.interpolate, mode="bicubic", align_corners=False
        )
        self.downscale_func = functools.partial(
            F.interpolate, mode="bicubic", align_corners=False
        )

    def forward(self, out, gt):
        h, w = gt.size()[2:]
        gt_up = self.upscale_func(gt, size=(h * self.scale, w * self.scale))
        gt_up_down = self.downscale_func(gt_up, size=gt.size()[2:])
        diff = torch.abs(gt - gt_up_down)
        diff_normal = mask_normal(diff)
        weight = self.weig_func2(diff_normal, self.alpha, self.gamma).detach()
        loss = torch.mean(weight * torch.abs(gt - out))
        return loss


def build_model():
    model = D2A2(args).cuda()
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
    return model


def build_train_loader():
    if args.dataset == "nyu":
        train_dataset = NYU_v2_dataset(
            root_dir=args.dataset_dir,
            scale=args.scale,
            augment=args.augment,
            input_size=args.input_size,
        )
    else:
        raise NotImplementedError("Dataset %s not found for training" % args.dataset)
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )


def build_test_loader():
    if args.dataset == "nyu":
        test_dataset = NYU_v2_dataset(
            root_dir=args.dataset_dir,
            scale=args.scale,
            train=False,
            augment=False,
            input_size=None,
        )
    elif args.dataset == "lu":
        test_dataset = LU_dataset(root_dir=args.dataset_dir, scale=args.scale)
    elif args.dataset == "middlebury":
        test_dataset = Middlebury_dataset(root_dir=args.dataset_dir, scale=args.scale)
    elif args.dataset == "rgbdd":
        test_dataset = RGBDD_dataset(root_dir=args.dataset_dir, scale=args.scale)
    else:
        raise NotImplementedError("Dataset %s not found" % args.dataset)
    return torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


def build_optimizer_scheduler(model):
    if args.pretrain_path is not None:
        load_state_dict_flexible(model, args.pretrain_path)
        optimizer = torch.optim.Adam(
            [{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=0.5,
        last_epoch=args.last_epoch,
    )
    return optimizer, scheduler


def build_criterion():
    if args.loss.lower() == "l1":
        return torch.nn.L1Loss()
    return MaskLoss(args.scale)


def main():
    setup_seed(20)
    mlflow.set_experiment(args.mlflow_experiment)

    model = build_model()
    train_loader = build_train_loader()
    test_loader = build_test_loader()
    optimizer, scheduler = build_optimizer_scheduler(model)
    criterion = build_criterion()

    run_kwargs = {}
    if args.mlflow_run_name is not None:
        run_kwargs["run_name"] = args.mlflow_run_name

    with mlflow.start_run(**run_kwargs) as run:
        mlflow.set_tag("pipeline", "train_then_test_best_model")
        mlflow.set_tag("run_id", run.info.run_id)

        trainer = MLFlowTrainer(
            args, model, optimizer, scheduler, criterion, train_loader, test_loader
        )
        train_summary = trainer.train()

        best_parameter_path = train_summary["best_parameter_path"]
        if best_parameter_path is None:
            best_parameter_path = train_summary["last_parameter_path"]
        if best_parameter_path is None:
            raise RuntimeError("No checkpoint was produced by training.")

        load_state_dict_flexible(model, best_parameter_path)
        mlflow.log_param("tested_checkpoint_path", best_parameter_path)

        tester = Tester(args, model, test_loader)
        test_rmse = tester.validate(
            mlflow_prefix="best_model_test",
            mlflow_step=train_summary["best_epoch"],
            log_mlflow=True,
            mlflow_artifact_path="best_model_test",
        )
        mlflow.log_metric("best_model_test_rmse_for_gate", float(test_rmse.mean()))

        print("MLflow run_id: %s" % run.info.run_id)
        print("best checkpoint: %s" % best_parameter_path)
        print("best_model_test_mean_rmse: %.10f" % float(test_rmse.mean()))


if __name__ == "__main__":
    main()
