"""
COSMEON - Climate Risk Intelligence Engine
Main entry point.

Usage:
  python main.py pipeline    → Run the full processing pipeline
  python main.py api         → Start the FastAPI server
  python main.py demo        → Run pipeline with demo data then start API
"""
import sys
import logging

from config.logging_config import setup_logging


def run_pipeline():
    """Run the full satellite data processing pipeline."""
    from pipeline import CosmeonPipeline

    pipeline = CosmeonPipeline()

    print("\n" + "=" * 60)
    print("  COSMEON Climate Risk Intelligence Engine")
    print("  Running full pipeline...")
    print("=" * 60 + "\n")

    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Region:              {results['region']['name']}")
    print(f"  Current detections:  {len(results['current_detections'])}")
    print(f"  Baseline detections: {len(results['baseline_detections'])}")
    print(f"  Change events:       {len(results['change_results'])}")
    print(f"  Risk assessments:    {len(results['risk_assessments'])}")
    print(f"  Duration:            {results['pipeline_duration_s']}s")

    if results['risk_assessments']:
        print("\n  Latest Risk Assessments:")
        for a in results['risk_assessments']:
            print(f"    [{a.risk_level:>8}] flood={a.flood_percentage*100:.1f}% "
                  f"confidence={a.confidence_score:.3f} "
                  f"change={a.change_type}")

    if results['change_results']:
        print("\n  Change Detection Results:")
        for c in results['change_results']:
            print(f"    [{c.change_type:>15}] water_change={c.water_change_pct*100:+.2f}% "
                  f"new_flood={c.new_flood_pixels}px "
                  f"area={c.affected_area_km2:.2f}km²")

    print("=" * 60 + "\n")
    return results


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import API_HOST, API_PORT

    setup_logging()
    print(f"\n  Starting COSMEON API server at http://{API_HOST}:{API_PORT}")
    print(f"  API docs: http://localhost:{API_PORT}/docs\n")

    uvicorn.run(
        "api.routes:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )


def run_demo():
    """Run pipeline then start API for demo."""
    results = run_pipeline()
    print("\n  Now starting API server for interactive exploration...\n")
    run_api()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [pipeline|api|demo]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "pipeline":
        run_pipeline()
    elif command == "api":
        run_api()
    elif command == "demo":
        run_demo()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [pipeline|api|demo]")
        sys.exit(1)
