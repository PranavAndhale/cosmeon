"""
COSMEON Model Training Script.

Standalone script to train the flood prediction model on real
historical weather and river discharge data from Open-Meteo APIs.

Usage:
    python train_model.py              # Train with real data (default)
    python train_model.py --synthetic  # Train with synthetic data only
    python train_model.py --regions 3  # Train with first N regions

Output:
    - Trained model saved to data/processed/flood_predictor.joblib
    - Scaler saved to data/processed/flood_scaler.joblib
    - Training metrics printed to console and logged
"""
import sys
import os
import argparse
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config.logging_config import setup_logging
from processing.predictor import FloodPredictor


def main():
    parser = argparse.ArgumentParser(description="Train the COSMEON flood prediction model")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data only (no API calls)",
    )
    parser.add_argument(
        "--regions", type=int, default=22,
        help="Number of regions to train on (default: 22)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("cosmeon.train")

    print("\n" + "=" * 60)
    print("  COSMEON Flood Prediction Model Training")
    print("=" * 60 + "\n")

    predictor = FloodPredictor()

    # Define training regions (22 global flood-prone locations)
    all_regions = [
        # South Asia — monsoon-driven, excellent GloFAS coverage
        {"name": "Bihar, India",          "lat": 26.0,   "lon": 85.5,    "elevation": 55},
        {"name": "Dhaka, Bangladesh",     "lat": 23.75,  "lon": 90.4,    "elevation": 8},
        {"name": "Navi Mumbai, India",    "lat": 19.1,   "lon": 73.0,    "elevation": 14},
        {"name": "Kolkata, India",        "lat": 22.6,   "lon": 88.4,    "elevation": 6},
        {"name": "Assam, India",          "lat": 26.2,   "lon": 92.5,    "elevation": 55},
        {"name": "Sylhet, Bangladesh",    "lat": 24.9,   "lon": 91.9,    "elevation": 15},
        # Southeast Asia
        {"name": "Jakarta, Indonesia",    "lat": -6.2,   "lon": 106.8,   "elevation": 8},
        {"name": "Bangkok, Thailand",     "lat": 13.75,  "lon": 100.5,   "elevation": 2},
        {"name": "Ho Chi Minh City",      "lat": 10.8,   "lon": 106.7,   "elevation": 5},
        {"name": "Manila, Philippines",   "lat": 14.6,   "lon": 121.0,   "elevation": 15},
        # Europe — Rhine/Danube basin
        {"name": "Bremen, Germany",       "lat": 53.1,   "lon": 8.8,     "elevation": 12},
        {"name": "Rotterdam, Netherlands","lat": 51.9,   "lon": 4.5,     "elevation": 0},
        {"name": "Budapest, Hungary",     "lat": 47.5,   "lon": 19.0,    "elevation": 105},
        {"name": "Venice, Italy",         "lat": 45.4,   "lon": 12.3,    "elevation": 1},
        # Americas
        {"name": "São Paulo, Brazil",     "lat": -23.55, "lon": -46.6,   "elevation": 760},
        {"name": "Manaus, Brazil",        "lat": -3.1,   "lon": -60.0,   "elevation": 92},
        {"name": "New Orleans, USA",      "lat": 29.95,  "lon": -90.07,  "elevation": 0},
        {"name": "Houston, USA",          "lat": 29.76,  "lon": -95.37,  "elevation": 13},
        # East Asia — Yangtze basin
        {"name": "Wuhan, China",          "lat": 30.6,   "lon": 114.3,   "elevation": 23},
        {"name": "Chongqing, China",      "lat": 29.6,   "lon": 106.5,   "elevation": 259},
        # Africa
        {"name": "Khartoum, Sudan",       "lat": 15.55,  "lon": 32.53,   "elevation": 380},
        {"name": "Lagos, Nigeria",        "lat": 6.45,   "lon": 3.4,     "elevation": 2},
    ]

    regions = all_regions[:args.regions]

    if args.synthetic:
        print("  Mode: SYNTHETIC data training")
        print(f"  Samples: 500\n")
        synthetic_data = predictor._generate_synthetic_data(500)
        metrics = predictor.train(synthetic_data)
        metrics["data_source"] = "synthetic"
    else:
        print(f"  Mode: REAL data training (Open-Meteo APIs)")
        print(f"  Regions: {len(regions)}")
        for r in regions:
            print(f"    → {r['name']} (lat={r['lat']}, lon={r['lon']}, elev={r['elevation']}m)")
        print()
        metrics = predictor.train_on_real_data(regions=regions)

    # Print results
    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    print(f"  Data source:     {metrics.get('data_source', 'mixed')}")
    print(f"  Total samples:   {metrics.get('total_samples', 'N/A')}")
    print(f"  Train samples:   {metrics.get('train_samples', 'N/A')}")
    print(f"  Test samples:    {metrics.get('test_samples', 'N/A')}")
    print(f"  Train accuracy:  {metrics.get('train_accuracy', 'N/A')}")
    print(f"  Test accuracy:   {metrics.get('test_accuracy', 'N/A')}")
    print(f"  CV mean (5-fold): {metrics.get('cv_mean', 'N/A')} ± {metrics.get('cv_std', 'N/A')}")

    if "cv_scores" in metrics:
        print(f"  CV per-fold:     {metrics['cv_scores']}")

    if "label_distribution" in metrics:
        print(f"\n  Label distribution:")
        for label, count in metrics["label_distribution"].items():
            total = metrics.get("total_samples", 1)
            print(f"    {label:>8}: {count} ({count/total*100:.1f}%)")

    if "feature_importances" in metrics:
        print(f"\n  Feature importances (top 5):")
        sorted_feats = sorted(
            metrics["feature_importances"].items(),
            key=lambda x: x[1], reverse=True,
        )
        for name, imp in sorted_feats[:5]:
            bar = "█" * int(imp * 50)
            print(f"    {name:<20} {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("  Model saved to: data/processed/flood_predictor.joblib")
    print("  Scaler saved to: data/processed/flood_scaler.joblib")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    main()
