import argparse
import traceback
from collections import Counter

from datasets.nyu import NYU_v2_dataset


def check_dataset(dataset, name):
    print(f"\n========== checking {name} ==========")
    print("dataset length:", len(dataset))

    ok = 0
    bad = 0
    error_counter = Counter()

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            ok += 1

            if idx % 100 == 0:
                print(f"[{name}] checked {idx}/{len(dataset)}")

        except Exception as e:
            bad += 1
            err_name = type(e).__name__
            error_counter[err_name] += 1

            print("\n" + "=" * 80)
            print(f"[BAD SAMPLE] dataset={name}, index={idx}")
            print("error type:", err_name)
            print("error:", repr(e))
            # traceback.print_exc()
            print("=" * 80)

    print(f"\n========== summary: {name} ==========")
    print("ok:", ok)
    print("bad:", bad)
    print("errors:", dict(error_counter))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--input_size", type=int, nargs="*", default=None)
    args = parser.parse_args()

    train_dataset = NYU_v2_dataset(
        root_dir=args.dataset_dir,
        scale=args.scale,
        augment=args.augment,
        input_size=args.input_size,
    )

    test_dataset = NYU_v2_dataset(
        root_dir=args.dataset_dir,
        scale=args.scale,
        train=False,
        augment=False,
        input_size=None,
    )

    check_dataset(train_dataset, "train")
    check_dataset(test_dataset, "test")


if __name__ == "__main__":
    main()
