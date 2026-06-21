import argparse
import json
import os
import tempfile

import mlflow.pyfunc
import pandas as pd
import torch

from datasets.nyu import NYU_v2_dataset


def tensor_summary(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={float(tensor.min()):.6f}, max={float(tensor.max()):.6f}, "
        f"mean={float(tensor.mean()):.6f}"
    )


def write_pyfunc_inputs(sample, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "rgb_path": os.path.abspath(os.path.join(output_dir, "rgb.pt")),
        "depth_path": os.path.abspath(os.path.join(output_dir, "depth_lr.pt")),
        "mde_path": os.path.abspath(os.path.join(output_dir, "mde.pt")),
    }

    torch.save(sample["guidance"], paths["rgb_path"])
    torch.save(sample["target"], paths["depth_path"])
    torch.save(sample["mde"], paths["mde_path"])

    return paths


def write_invocation_json(paths, output_dir):
    request = {
        "columns": ["rgb_path", "depth_path", "mde_path"],
        "data": [[paths["rgb_path"], paths["depth_path"], paths["mde_path"]]],
    }
    request_path = os.path.abspath(os.path.join(output_dir, "invocation.json"))
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f)
    return request_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        default=(
            "mlruns/3/1d0b79e73e70467f8e02fd6a1a98ff6d/"
            "artifacts/best_model_pyfunc"
        ),
        help="MLflow pyfunc model URI or local model path.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/NYUv2",
        help="NYUv2 root dir. In this repo, datasets/NYUv2 is the valid copy.",
    )
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split used to fetch the sample.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory for generated .pt inputs. If omitted, a temporary "
            "directory is used and removed after prediction."
        ),
    )
    parser.add_argument(
        "--write-request-json",
        action="store_true",
        help="Also write an invocation.json file for curl or mlflow models serve.",
    )
    args = parser.parse_args()

    dataset = NYU_v2_dataset(
        root_dir=args.dataset_dir,
        scale=args.scale,
        train=args.split == "train",
        augment=False,
        input_size=None,
    )
    sample = dataset[args.index]

    print(f"dataset_dir: {args.dataset_dir}")
    print(f"split: {args.split}, index: {args.index}, scale: {args.scale}")
    tensor_summary("guidance", sample["guidance"])
    tensor_summary("target_lr_depth", sample["target"])
    tensor_summary("gt_depth", sample["gt"])
    tensor_summary("mde", sample["mde"])

    context = tempfile.TemporaryDirectory(prefix="d2a2_pyfunc_nyu_")
    input_dir = args.input_dir or context.name

    try:
        paths = write_pyfunc_inputs(sample, input_dir)
        print("pyfunc input paths:")
        for key, value in paths.items():
            print(f"  {key}: {value}")

        if args.write_request_json:
            request_path = write_invocation_json(paths, input_dir)
            print(f"invocation_json: {request_path}")

        model = mlflow.pyfunc.load_model(args.model_uri)
        output = model.predict(pd.DataFrame([paths]))
        output_tensor = torch.tensor(output[0])
        tensor_summary("prediction", output_tensor)
    finally:
        if args.input_dir is None:
            context.cleanup()


if __name__ == "__main__":
    main()
