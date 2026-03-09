"""
Phase 3: Change Detection - Compare historical and recent satellite data.

Detects changes in water extent between two time periods to identify
new flooding events or environmental changes.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from shapely.geometry import shape, mapping
from skimage import morphology, measure

from config.settings import FLOOD_CHANGE_THRESHOLD

logger = logging.getLogger("cosmeon.processing.change")


@dataclass
class ChangeDetectionResult:
    """Result of change detection between two time periods."""
    baseline_id: str
    current_id: str
    baseline_date: str
    current_date: str
    baseline_water_pct: float
    current_water_pct: float
    water_change_pct: float
    new_flood_pixels: int
    receded_pixels: int
    change_type: str  # "flood_increase", "flood_decrease", "stable"
    change_mask: np.ndarray = field(repr=False)
    affected_area_km2: float = 0.0
    change_polygons: list = field(default_factory=list)
    bbox: list = field(default_factory=list)
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ChangeDetector:
    """Detects changes between baseline and current flood maps."""

    def __init__(self, change_threshold: float = None):
        self.change_threshold = change_threshold or FLOOD_CHANGE_THRESHOLD
        logger.info("ChangeDetector initialized | threshold=%.2f", self.change_threshold)

    def compute_change_mask(
        self,
        baseline_mask: np.ndarray,
        current_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute change mask between baseline and current water masks.

        Returns:
            Change mask with values:
              0 = no change (both dry or both wet)
              1 = new flooding (was dry, now wet)
              2 = water recession (was wet, now dry)
        """
        if baseline_mask.shape != current_mask.shape:
            from skimage.transform import resize
            baseline_mask = resize(
                baseline_mask, current_mask.shape, order=0, preserve_range=True
            ).astype(np.uint8)

        change = np.zeros_like(current_mask, dtype=np.uint8)
        change[(baseline_mask == 0) & (current_mask == 1)] = 1  # new flooding
        change[(baseline_mask == 1) & (current_mask == 0)] = 2  # recession
        return change

    def compute_ndwi_difference(
        self,
        baseline_ndwi: np.ndarray,
        current_ndwi: np.ndarray,
    ) -> np.ndarray:
        """Compute NDWI difference map (current - baseline)."""
        if baseline_ndwi.shape != current_ndwi.shape:
            from skimage.transform import resize
            baseline_ndwi = resize(
                baseline_ndwi, current_ndwi.shape, order=1, preserve_range=True
            )
        diff = current_ndwi - baseline_ndwi
        return np.nan_to_num(diff, nan=0.0)

    def extract_change_polygons(
        self,
        change_mask: np.ndarray,
        transform=None,
        min_area_pixels: int = 100,
    ) -> list[dict]:
        """
        Extract change regions as GeoJSON-like polygon features.

        Args:
            change_mask: Binary mask of changed areas (1 = new flooding)
            transform: Rasterio affine transform for georeferencing
            min_area_pixels: Minimum region size to include

        Returns:
            List of dicts with geometry and properties
        """
        new_flood = (change_mask == 1).astype(np.uint8)
        labeled = measure.label(new_flood)
        regions = measure.regionprops(labeled)

        polygons = []
        for region in regions:
            if region.area < min_area_pixels:
                continue

            # Get bounding box as simple polygon
            minr, minc, maxr, maxc = region.bbox
            polygon = {
                "type": "Feature",
                "properties": {
                    "area_pixels": region.area,
                    "area_km2": round(region.area * 100 / 1e6, 4),  # 10m resolution
                    "centroid_row": float(region.centroid[0]),
                    "centroid_col": float(region.centroid[1]),
                    "change_type": "new_flooding",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [int(minc), int(minr)],
                        [int(maxc), int(minr)],
                        [int(maxc), int(maxr)],
                        [int(minc), int(maxr)],
                        [int(minc), int(minr)],
                    ]],
                },
            }

            # Apply georeferencing if transform is available
            if transform is not None:
                coords = polygon["geometry"]["coordinates"][0]
                geo_coords = []
                for c, r in coords:
                    x, y = transform * (c, r)
                    geo_coords.append([round(x, 6), round(y, 6)])
                polygon["geometry"]["coordinates"] = [geo_coords]

            polygons.append(polygon)

        logger.info("Extracted %d change polygons (min area=%d px)", len(polygons), min_area_pixels)
        return polygons

    def classify_change(self, water_change_pct: float) -> str:
        """Classify the type of change detected."""
        if water_change_pct > self.change_threshold:
            return "flood_increase"
        elif water_change_pct < -self.change_threshold:
            return "flood_decrease"
        return "stable"

    def detect(
        self,
        baseline_result,
        current_result,
        transform=None,
    ) -> ChangeDetectionResult:
        """
        Run change detection between baseline and current flood detection results.

        Args:
            baseline_result: FloodDetectionResult from historical period
            current_result: FloodDetectionResult from recent period
            transform: Optional rasterio transform for georeferencing

        Returns:
            ChangeDetectionResult with change analysis
        """
        logger.info(
            "=== Running change detection: %s vs %s ===",
            baseline_result.item_id, current_result.item_id,
        )

        # Compute change mask
        change_mask = self.compute_change_mask(
            baseline_result.water_mask,
            current_result.water_mask,
        )

        # Calculate statistics
        new_flood_pixels = int(np.sum(change_mask == 1))
        receded_pixels = int(np.sum(change_mask == 2))
        water_change_pct = current_result.flood_percentage - baseline_result.flood_percentage

        # Classify change type
        change_type = self.classify_change(water_change_pct)

        # Extract change polygons
        change_polygons = self.extract_change_polygons(change_mask, transform)

        # Estimate affected area
        affected_area = round((new_flood_pixels * 100) / 1e6, 4)  # 10m resolution

        result = ChangeDetectionResult(
            baseline_id=baseline_result.item_id,
            current_id=current_result.item_id,
            baseline_date=baseline_result.timestamp,
            current_date=current_result.timestamp,
            baseline_water_pct=baseline_result.flood_percentage,
            current_water_pct=current_result.flood_percentage,
            water_change_pct=round(water_change_pct, 4),
            new_flood_pixels=new_flood_pixels,
            receded_pixels=receded_pixels,
            change_type=change_type,
            change_mask=change_mask,
            affected_area_km2=affected_area,
            change_polygons=change_polygons,
            bbox=current_result.bbox,
        )

        logger.info(
            "Change result: type=%s | water_change=%.2f%% | new_flood=%d px | receded=%d px | area=%.2f km²",
            change_type, water_change_pct * 100,
            new_flood_pixels, receded_pixels, affected_area,
        )

        return result
