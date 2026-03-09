"""
Phase 1: Satellite Data Ingestion via Microsoft Planetary Computer STAC API.

Fetches Sentinel-2 L2A imagery for specified regions and date ranges.
Supports cloud-optimized GeoTIFF (COG) access — no bulk downloads needed.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

from config.settings import (
    DEFAULT_BBOX,
    DEFAULT_REGION_NAME,
    MAX_CLOUD_COVER,
    PC_STAC_URL,
    RAW_DIR,
    SENTINEL2_COLLECTION,
)

logger = logging.getLogger("cosmeon.ingestion")


class SatelliteFetcher:
    """Fetches satellite imagery from Microsoft Planetary Computer."""

    def __init__(self):
        self.catalog = pystac_client.Client.open(
            PC_STAC_URL,
            modifier=pc.sign_inplace,
        )
        logger.info("Connected to Planetary Computer STAC API")

    def search_items(
        self,
        bbox: list[float] = None,
        date_range: str = None,
        collection: str = None,
        max_cloud_cover: int = None,
        limit: int = 5,
    ) -> list:
        """Search for satellite imagery items matching criteria."""
        bbox = bbox or DEFAULT_BBOX
        collection = collection or SENTINEL2_COLLECTION
        max_cloud_cover = max_cloud_cover if max_cloud_cover is not None else MAX_CLOUD_COVER

        if date_range is None:
            end = datetime.utcnow()
            start = end - timedelta(days=90)
            date_range = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        logger.info(
            "Searching %s | bbox=%s | dates=%s | cloud<=%d%%",
            collection, bbox, date_range, max_cloud_cover,
        )

        search = self.catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=limit,
            sortby=[{"field": "datetime", "direction": "desc"}],
        )

        items = list(search.items())
        logger.info("Found %d items", len(items))
        return items

    def fetch_bands(
        self,
        item,
        bands: list[str] = None,
        bbox: list[float] = None,
        resolution: int = 10,
    ) -> dict:
        """
        Fetch specific bands from a STAC item as numpy arrays.

        Args:
            item: PYSTAC item
            bands: List of band names (default: ["B03", "B08"] for NDWI)
            bbox: Bounding box to clip [west, south, east, north]
            resolution: Target resolution in meters

        Returns:
            dict with band_name -> numpy array mapping, plus metadata
        """
        bands = bands or ["B03", "B08", "B04", "SCL"]
        bbox = bbox or DEFAULT_BBOX

        result = {
            "item_id": item.id,
            "datetime": str(item.datetime),
            "bbox": bbox,
            "bands": {},
            "transform": None,
            "crs": None,
            "shape": None,
        }

        for band_name in bands:
            if band_name not in item.assets:
                logger.warning("Band %s not found in item %s", band_name, item.id)
                continue

            asset = item.assets[band_name]
            href = asset.href

            logger.info("Reading band %s from %s", band_name, item.id)

            try:
                with rasterio.open(href) as src:
                    # Reproject bbox from EPSG:4326 to the raster's CRS
                    from pyproj import Transformer
                    src_crs = str(src.crs)
                    if src_crs != "EPSG:4326":
                        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
                        x_min, y_min = transformer.transform(bbox[0], bbox[1])
                        x_max, y_max = transformer.transform(bbox[2], bbox[3])
                        reprojected_bbox = [x_min, y_min, x_max, y_max]
                    else:
                        reprojected_bbox = bbox

                    # Clip to raster bounds to avoid zero-size windows
                    src_bounds = src.bounds
                    clipped_bbox = [
                        max(reprojected_bbox[0], src_bounds.left),
                        max(reprojected_bbox[1], src_bounds.bottom),
                        min(reprojected_bbox[2], src_bounds.right),
                        min(reprojected_bbox[3], src_bounds.top),
                    ]

                    # Check if there's actual overlap
                    if clipped_bbox[0] >= clipped_bbox[2] or clipped_bbox[1] >= clipped_bbox[3]:
                        logger.warning("No overlap between bbox and tile %s for band %s", item.id, band_name)
                        continue

                    window = from_bounds(*clipped_bbox, transform=src.transform)

                    # Ensure window has positive dimensions
                    if window.width < 1 or window.height < 1:
                        logger.warning("Window too small for band %s in %s", band_name, item.id)
                        continue

                    data = src.read(1, window=window).astype(np.float32)

                    if data.size == 0:
                        logger.warning("Empty data for band %s in %s", band_name, item.id)
                        continue

                    if result["transform"] is None:
                        result["transform"] = src.window_transform(window)
                        result["crs"] = str(src.crs)
                        result["shape"] = data.shape

                    # Resize to match first band if shapes differ
                    if result["shape"] is not None and data.shape != result["shape"]:
                        from skimage.transform import resize
                        data = resize(data, result["shape"], order=1, preserve_range=True).astype(np.float32)

                    result["bands"][band_name] = data
                    logger.info(
                        "Band %s: shape=%s, min=%.2f, max=%.2f",
                        band_name, data.shape, np.nanmin(data), np.nanmax(data),
                    )
            except Exception as e:
                logger.error("Error reading band %s from %s: %s", band_name, item.id, e)
                continue

        return result

    def save_metadata(self, item, output_dir: Path = None) -> Path:
        """Save item metadata as JSON for audit trail."""
        output_dir = output_dir or RAW_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "id": item.id,
            "datetime": str(item.datetime),
            "bbox": list(item.bbox),
            "geometry": mapping(box(*item.bbox)),
            "properties": {
                k: v for k, v in item.properties.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
            "assets": list(item.assets.keys()),
            "collection": item.collection_id,
            "ingested_at": datetime.utcnow().isoformat(),
        }

        filepath = output_dir / f"{item.id}_metadata.json"
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Saved metadata to %s", filepath)
        return filepath

    def ingest(
        self,
        bbox: list[float] = None,
        date_range: str = None,
        bands: list[str] = None,
        limit: int = 3,
    ) -> list[dict]:
        """
        Full ingestion pipeline: search -> fetch bands -> save metadata.

        Returns list of band data dicts ready for processing.
        """
        logger.info("=== Starting ingestion pipeline ===")
        items = self.search_items(bbox=bbox, date_range=date_range, limit=limit)

        if not items:
            logger.warning("No items found for given criteria")
            return []

        results = []
        for item in items:
            try:
                band_data = self.fetch_bands(item, bands=bands, bbox=bbox)
                # Only keep results that have at least B03 and B08 (needed for NDWI)
                if "B03" in band_data["bands"] and "B08" in band_data["bands"]:
                    self.save_metadata(item)
                    results.append(band_data)
                    logger.info("Successfully ingested item %s", item.id)
                else:
                    logger.warning("Item %s missing required bands (B03/B08), skipping", item.id)
            except Exception as e:
                logger.error("Failed to ingest item %s: %s", item.id, str(e))
                continue

        logger.info("=== Ingestion complete: %d/%d items processed ===", len(results), len(items))
        return results
