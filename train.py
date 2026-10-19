from __future__ import annotations

import argparse
import json

from lung_cancer_pipeline import (
    load_tabular_dataset,
    train_image_model,
    train_tabular_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local lung cancer models.")
    parser.add_argument("--tabular-sample", type=int, default=0, help="Rows to use; 0 uses the complete CSV.")
    parser.add_argument("--image-sample-per-class", type=int, default=50)
    parser.add_argument("--image-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-fine-tune", action="store_true")
    args = parser.parse_args()

    tabular = train_tabular_model(sample_size=args.tabular_sample or len(load_tabular_dataset()))
    image = train_image_model(
        sample_per_class=args.image_sample_per_class,
        epochs=args.image_epochs,
        batch_size=args.batch_size,
        fine_tune=not args.no_fine_tune,
    )
    print(json.dumps({"tabular": tabular["metrics"], "image": image["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
