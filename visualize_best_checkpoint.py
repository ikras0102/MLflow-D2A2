import os

from option import args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from datasets import *
from utils import Tester, setup_seed


if args.scale == 4:
    from models.D2A2_depthanything import D2A2
elif args.scale == 8:
    from models.D2A2_depthanything_L_scale8 import D2A2
elif args.scale == 16:
    from models.D2A2_depthanything_L_scale16 import D2A2
else:
    raise NotImplementedError("Scale %s not found" % args.scale)


def _as_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def load_state_dict_flexible(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
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


def build_model():
    model = D2A2(args).cuda()

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    return model


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


def main():
    if args.net_path is None:
        raise ValueError("Please set --net_path to the checkpoint path.")

    args.save = True
    setup_seed(20)

    model = build_model()
    load_state_dict_flexible(model, args.net_path)
    print("model done")

    test_loader = build_test_loader()
    print("dataloader done")

    tester = Tester(args, model, test_loader)
    print("tester done")

    tester.validate()

    print("tester.result_root:", tester.result_root)
    print("tester.save_hotmap_root:", tester.save_hotmap_root)


if __name__ == "__main__":
    main()
