"""
Standalone U-Net Training Script for Sen1Floods11.

Usage:
    python train_unet.py                    # Train on synthetic data (demo)
    python train_unet.py --samples 200      # Train on 200 samples
    python train_unet.py --epochs 30        # Train for 30 epochs

For real data, download Sen1Floods11 to data/sen1floods11/S1Hand/ and data/sen1floods11/LabelHand/
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_unet")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Sen1Floods11")
    parser.add_argument("--samples", type=int, default=300, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from ml.sen1floods11_loader import Sen1Floods11Loader
    from ml.unet_model import FloodModelManager

    # Load dataset
    loader = Sen1Floods11Loader()
    print(f"\n{'='*60}")
    print("COSMEON U-Net Training — Sen1Floods11")
    print(f"{'='*60}")

    info = loader.get_dataset_info()
    print(f"Dataset: {info['name']}")
    print(f"Local data: {'Available' if info['local_data_available'] else 'Not found (using synthetic)'}")

    images, masks = loader.load_dataset(max_samples=args.samples)
    if not images:
        print("ERROR: No training data available")
        return

    print(f"Loaded {len(images)} images ({images[0].shape})")
    print(f"Flood pixel ratio: {np.mean([m.mean() for m in masks])*100:.1f}%")
    print(f"\nTraining: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    # Train
    manager = FloodModelManager(in_channels=2)
    metrics = manager.train(
        images=images,
        masks=masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Print results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import numpy as np
    main()
