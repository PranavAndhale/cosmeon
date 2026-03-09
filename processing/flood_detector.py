"""
Phase 2: Flood Detection using NDWI (Normalized Difference Water Index).

Uses Sentinel-2 bands:
  - B03 (Green): 560nm
  - B08 (NIR): 842nm

NDWI = (Green - NIR) / (Green + NIR)
Water pixels have NDWI > threshold (typically 0.3).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from skimage import morphology

from config.settings import NDWI_THRESHOLD, RISK_THRESHOLDS

logger = logging.getLogger("cosmeon.processing.flood")


@dataclass
class FloodDetectionResult:
    """Result of flood detection analysis on a single image."""
    item_id: str
    timestamp: str
    total_pixels: int
    water_pixels: int
    flood_percentage: float
    risk_level: str
    ndwi_map: np.ndarray = field(repr=False)
    water_mask: np.ndarray = field(repr=False)
    confidence_score: float = 0.0
    bbox: list = field(default_factory=list)
    area_km2: float = 0.0
    flood_area_km2: float = 0.0
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FloodDetector:
    """Detects flood-affected regions using NDWI thresholding."""

    def __init__(self, ndwi_threshold: float = None):
        self.ndwi_threshold = ndwi_threshold or NDWI_THRESHOLD
        logger.info("FloodDetector initialized | threshold=%.2f", self.ndwi_threshold)

    def compute_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Compute NDWI from Green and NIR bands.

        NDWI = (Green - NIR) / (Green + NIR)
        Range: [-1, 1], water > 0.3 typically
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi = np.clip(ndwi, -1, 1)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
        return ndwi

    def apply_cloud_mask(self, data: np.ndarray, scl: np.ndarray) -> np.ndarray:
        """
        Mask out cloud/shadow pixels using Sentinel-2 Scene Classification Layer.

        SCL values to mask:
          3 = cloud shadow, 8 = cloud medium prob, 9 = cloud high prob, 10 = thin cirrus
        """
        cloud_mask = np.isin(scl, [3, 8, 9, 10])
        masked = data.copy()
        masked[cloud_mask] = np.nan
        cloud_pct = np.sum(cloud_mask) / cloud_mask.size * 100
        logger.info("Cloud mask applied: %.1f%% pixels masked", cloud_pct)
        return masked

    def create_water_mask(self, ndwi: np.ndarray) -> np.ndarray:
        """Create binary water mask from NDWI with morphological cleanup."""
        water = (ndwi > self.ndwi_threshold).astype(np.uint8)

        # Remove small noise (objects < 50 pixels)
        water_cleaned = morphology.remove_small_objects(
            water.astype(bool), min_size=50
        ).astype(np.uint8)

        # Fill small holes
        water_cleaned = morphology.remove_small_holes(
            water_cleaned.astype(bool), area_threshold=25
        ).astype(np.uint8)

        return water_cleaned

    def classify_risk(self, flood_percentage: float) -> str:
        """Classify flood risk level based on percentage of area flooded."""
        for level, threshold in sorted(
            RISK_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if flood_percentage >= threshold:
                return level
        return "LOW"

    def compute_confidence(
        self, ndwi: np.ndarray, water_mask: np.ndarray, scl: np.ndarray = None
    ) -> float:
        """
        Compute confidence score for flood detection (0.0 - 1.0).

        Based on:
          - Clear pixel ratio (less clouds = higher confidence)
          - NDWI strength (stronger signal = higher confidence)
          - Spatial coherence (less fragmented = higher confidence)
        """
        scores = []

        # Cloud coverage penalty
        if scl is not None:
            clear_ratio = 1.0 - np.mean(np.isin(scl, [3, 8, 9, 10]))
            scores.append(clear_ratio)
        else:
            scores.append(0.7)

        # NDWI signal strength (mean NDWI of detected water pixels)
        if np.any(water_mask):
            mean_water_ndwi = np.nanmean(ndwi[water_mask.astype(bool)])
            signal_score = min(mean_water_ndwi / 0.5, 1.0)
            scores.append(max(signal_score, 0.0))
        else:
            scores.append(0.5)

        # Spatial coherence (ratio of largest component to total water)
        labeled = morphology.label(water_mask)
        if labeled.max() > 0:
            component_sizes = np.bincount(labeled.ravel())[1:]
            coherence = component_sizes.max() / component_sizes.sum()
            scores.append(coherence)
        else:
            scores.append(0.5)

        return round(float(np.mean(scores)), 3)

    def estimate_area(self, pixel_count: int, resolution_m: float = 10.0) -> float:
        """Convert pixel count to area in km^2."""
        pixel_area_m2 = resolution_m * resolution_m
        return round(pixel_count * pixel_area_m2 / 1e6, 4)

    def detect(self, band_data: dict) -> FloodDetectionResult:
        """
        Run flood detection on ingested band data.

        Args:
            band_data: dict from SatelliteFetcher.fetch_bands() with keys:
                - bands: {"B03": array, "B08": array, "SCL": array}
                - item_id, datetime, bbox, shape

        Returns:
            FloodDetectionResult with all detection metrics
        """
        logger.info("=== Running flood detection on %s ===", band_data["item_id"])

        green = band_data["bands"]["B03"]
        nir = band_data["bands"]["B08"]
        scl = band_data["bands"].get("SCL")

        # Apply cloud mask if SCL available
        if scl is not None:
            # Resize SCL to match other bands if needed (SCL is 20m, others 10m)
            if scl.shape != green.shape:
                from skimage.transform import resize
                scl = resize(
                    scl, green.shape, order=0, preserve_range=True
                ).astype(scl.dtype)
            green = self.apply_cloud_mask(green, scl)
            nir = self.apply_cloud_mask(nir, scl)

        # Compute NDWI
        ndwi = self.compute_ndwi(green, nir)
        logger.info(
            "NDWI computed: min=%.3f, max=%.3f, mean=%.3f",
            np.nanmin(ndwi), np.nanmax(ndwi), np.nanmean(ndwi),
        )

        # Create water mask
        water_mask = self.create_water_mask(ndwi)

        # Calculate statistics
        valid_pixels = np.sum(~np.isnan(ndwi))
        water_pixels = int(np.sum(water_mask))
        flood_pct = water_pixels / max(valid_pixels, 1)

        # Classify risk
        risk_level = self.classify_risk(flood_pct)

        # Compute confidence
        confidence = self.compute_confidence(ndwi, water_mask, scl)

        # Estimate areas
        total_area = self.estimate_area(valid_pixels)
        flood_area = self.estimate_area(water_pixels)

        result = FloodDetectionResult(
            item_id=band_data["item_id"],
            timestamp=band_data["datetime"],
            total_pixels=int(valid_pixels),
            water_pixels=water_pixels,
            flood_percentage=round(flood_pct, 4),
            risk_level=risk_level,
            ndwi_map=ndwi,
            water_mask=water_mask,
            confidence_score=confidence,
            bbox=band_data.get("bbox", []),
            area_km2=total_area,
            flood_area_km2=flood_area,
        )

        logger.info(
            "Detection result: %d/%d water pixels (%.2f%%) | risk=%s | confidence=%.3f | area=%.2f km²",
            water_pixels, valid_pixels, flood_pct * 100,
            risk_level, confidence, flood_area,
        )

        return result
