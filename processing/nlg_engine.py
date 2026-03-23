"""
Phase 10A: Natural Language Generation (NLG) Engine.

Generates human-readable executive summaries and insights from flood risk
analysis data. Uses template-based generation with optional OpenAI GPT
integration for richer narratives.
"""
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.nlg")

# In-memory cache for generated summaries (region_id -> {hash, text, generated_at})
_nlg_cache: dict[int, dict] = {}


class NLGEngine:
    """Generates natural language summaries from structured analysis data."""

    def __init__(self):
        self.gemini_model = None
        self.gemini_client = None
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.gemini_model = "gemini-2.0-flash"
                logger.info("NLG Engine initialized with Gemini 2.0 Flash")
            except ImportError:
                logger.warning("google-genai not installed — using template-based NLG")
            except Exception as e:
                logger.warning("Gemini init failed (%s) — using template-based NLG", e)
        else:
            logger.info("NLG Engine initialized (template-based, no GEMINI_API_KEY set)")

    def _data_hash(self, data: dict) -> str:
        """Create a hash of analysis data to detect changes for cache invalidation."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]

    def generate_executive_summary(
        self,
        region_name: str,
        risk_data: dict,
        prediction_data: Optional[dict] = None,
        detection_data: Optional[dict] = None,
        validation_data: Optional[dict] = None,
        external_factors: Optional[dict] = None,
        region_id: Optional[int] = None,
    ) -> dict:
        """
        Generate an executive summary narrative for a region's flood analysis.

        Returns:
            {
                "narrative": str,
                "highlights": list[str],
                "risk_trend": str,
                "generated_at": str,
                "engine": "gpt-4" | "template"
            }
        """
        # Build combined data for hashing
        combined = {
            "region": region_name,
            "risk": risk_data,
            "prediction": prediction_data,
            "detection": detection_data,
        }

        # Check cache
        if region_id and region_id in _nlg_cache:
            cached = _nlg_cache[region_id]
            if cached["hash"] == self._data_hash(combined):
                logger.info("NLG cache hit for region %s", region_id)
                return cached["result"]

        # Try Gemini-based generation first
        if self.gemini_client and self.gemini_model:
            result = self._generate_with_gemini(
                region_name, risk_data, prediction_data,
                detection_data, validation_data, external_factors
            )
        else:
            result = self._generate_with_templates(
                region_name, risk_data, prediction_data,
                detection_data, validation_data, external_factors
            )

        # Cache result
        if region_id:
            _nlg_cache[region_id] = {
                "hash": self._data_hash(combined),
                "result": result,
                "cached_at": datetime.utcnow().isoformat(),
            }

        return result

    def generate_alert_description(self, alert_data: dict) -> str:
        """Generate a narrative description for an alert event."""
        risk = alert_data.get("risk_level", "UNKNOWN")
        region = alert_data.get("region_name", "the monitored area")
        flood_pct = alert_data.get("flood_percentage", 0)
        confidence = alert_data.get("confidence_score", 0)

        severity_map = {
            "CRITICAL": ("🚨", "Critical", "Immediate action is strongly recommended."),
            "HIGH": ("⚠️", "High", "Close monitoring and preparedness measures are advised."),
            "MEDIUM": ("📋", "Elevated", "Continued observation is warranted."),
            "LOW": ("✅", "Low", "No immediate action is required."),
        }
        emoji, severity, action = severity_map.get(risk, ("ℹ️", risk, "Review the latest data."))

        return (
            f"{emoji} **{severity} Flood Risk Alert — {region}** | "
            f"Flood coverage has reached {flood_pct:.1%} of the monitored area "
            f"(confidence: {confidence:.0%}). {action}"
        )

    def generate_trend_narrative(self, timeline_data: list[dict]) -> str:
        """Generate a trend narrative from historical risk assessments."""
        if not timeline_data:
            return "Insufficient historical data to determine trends."

        levels = [d.get("risk_level", "LOW") for d in timeline_data]
        risk_values = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        scores = [risk_values.get(l, 1) for l in levels]

        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:-3]) / max(len(scores) - 3, 1)
            if recent_avg > older_avg + 0.5:
                trend = "escalating"
                trend_detail = "Risk levels have been increasing in recent assessments."
            elif recent_avg < older_avg - 0.5:
                trend = "improving"
                trend_detail = "Risk levels show a downward trend in recent assessments."
            else:
                trend = "stable"
                trend_detail = "Risk levels have remained relatively consistent."
        else:
            trend = "insufficient_data"
            trend_detail = "More data points are needed for a reliable trend analysis."

        latest = timeline_data[-1] if timeline_data else {}
        latest_risk = latest.get("risk_level", "UNKNOWN")
        latest_date = latest.get("timestamp", "N/A")

        return (
            f"**Trend: {trend.title()}** — {trend_detail} "
            f"The most recent assessment ({latest_date[:10] if isinstance(latest_date, str) else 'N/A'}) "
            f"indicates a **{latest_risk}** risk level across {len(timeline_data)} data points."
        )

    def _generate_with_gemini(
        self,
        region_name: str,
        risk_data: dict,
        prediction_data: Optional[dict],
        detection_data: Optional[dict],
        validation_data: Optional[dict],
        external_factors: Optional[dict],
    ) -> dict:
        """Generate narrative using Gemini 1.5 Flash (free tier)."""
        try:
            fv = (prediction_data or {}).get("feature_values", {}) or {}
            cf = (prediction_data or {}).get("contributing_factors", {}) or {}
            data_block = json.dumps({
                "region": region_name,
                "risk_level": (prediction_data or {}).get("risk_level", risk_data.get("risk_level")),
                "flood_probability": (prediction_data or {}).get("probability", 0),
                "confidence": (prediction_data or {}).get("confidence", 0),
                "glofas_tier": cf.get("glofas_tier"),
                "glofas_source": cf.get("glofas_source"),
                "discharge_anomaly_sigma": fv.get("discharge_anomaly_sigma"),
                "precip_7d_mm": fv.get("precip_7d_mm"),
                "precip_anomaly_sigma": fv.get("precip_anomaly"),
                "soil_saturation": fv.get("soil_saturation"),
                "precip_compound_uplift": cf.get("precip_compound"),
                "soil_compound_uplift": cf.get("soil_compound"),
                "top_drivers": (prediction_data or {}).get("top_drivers", [])[:3],
            }, indent=2, default=str)

            prompt = (
                "You are a climate intelligence analyst for the Cosmeon flood monitoring platform. "
                "Generate a concise professional executive summary from the flood analysis data below. "
                "The prediction uses GloFAS v4 river discharge (T1=operational forecast, T4=ERA5 surrogate) "
                "as the primary signal, compounded with ERA5 precipitation anomaly and soil moisture. "
                "Mention the GloFAS tier, discharge anomaly, and compound signals. Use specific numbers. "
                "Respond ONLY with valid JSON — no markdown, no code fences — with exactly these keys: "
                '"narrative" (2 short paragraphs as a single string, use \\n\\n to separate), '
                '"highlights" (array of 3-5 concise strings), '
                '"risk_trend" (exactly one of: escalating, stable, improving, critical).\n\n'
                f"Data:\n{data_block}"
            )

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model, contents=prompt
            )
            raw = response.text.strip()
            # Extract JSON — Gemini sometimes wraps in ```json...``` or adds preamble text
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON object found in Gemini response: {raw[:200]}")
            content = json.loads(json_match.group(0))
            content["generated_at"] = datetime.utcnow().isoformat()
            content["engine"] = self.gemini_model
            return content

        except Exception as e:
            logger.warning("Gemini generation failed (%s: %s), falling back to templates", type(e).__name__, e)
            result = self._generate_with_templates(
                region_name, risk_data, prediction_data,
                detection_data, validation_data, external_factors
            )
            result["engine"] = "gemini-2.0-flash(fallback)"
            return result

    def _generate_with_templates(
        self,
        region_name: str,
        risk_data: dict,
        prediction_data: Optional[dict],
        detection_data: Optional[dict],
        validation_data: Optional[dict],
        external_factors: Optional[dict],
    ) -> dict:
        """Generate narrative using template-based string interpolation.

        prediction_data is expected to be the output of explain_prediction()
        (keys: risk_level, probability, confidence, contributing_factors, feature_values).
        """
        # --- Primary data source: explain_prediction output ---
        if prediction_data:
            # explain_prediction() format
            risk_level = prediction_data.get("risk_level") or risk_data.get("risk_level", "UNKNOWN")
            pred_prob  = prediction_data.get("probability", prediction_data.get("flood_probability", 0))
            pred_conf  = prediction_data.get("confidence", 0)
            cf         = prediction_data.get("contributing_factors", {}) or {}
            fv         = prediction_data.get("feature_values", {}) or {}
            tier       = cf.get("glofas_tier", "?")
            source     = cf.get("glofas_source", "GloFAS v4")
            precip_delta = cf.get("precip_compound", 0.0)
            soil_delta   = cf.get("soil_compound", 0.0)
            precip_7d    = fv.get("precip_7d_mm", 0.0)
            anomaly_sigma = fv.get("discharge_anomaly_sigma", 0.0)
            soil_sat     = fv.get("soil_saturation", 0.0)
            glofas_level = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][
                min(int(round(fv.get("glofas_flood_risk", 0))), 3)
            ] if fv else "UNKNOWN"
        else:
            risk_level = risk_data.get("risk_level", "UNKNOWN")
            pred_prob  = risk_data.get("confidence_score", 0)
            pred_conf  = risk_data.get("confidence_score", 0)
            cf = fv = {}
            tier = source = "?"
            precip_delta = soil_delta = precip_7d = anomaly_sigma = soil_sat = 0.0
            glofas_level = "UNKNOWN"

        severity_phrases = {
            "CRITICAL": ("demands immediate emergency response", "critically elevated"),
            "HIGH": ("warrants close monitoring and preparedness", "elevated"),
            "MEDIUM": ("requires ongoing observation", "moderate"),
            "LOW": ("presents minimal concern at this time", "low"),
        }
        phrase, adj = severity_phrases.get(risk_level, ("requires review", risk_level.lower()))

        # --- Paragraph 1: Current ML-assessed situation ---
        p1 = (
            f"The TieredFloodPredictor (GloFAS v4 + ERA5) has assessed **{region_name}** "
            f"as **{risk_level}** flood risk — a threat level that {phrase}. "
            f"The estimated flood probability is **{pred_prob:.0%}** with "
            f"**{pred_conf:.0%}** model confidence, based on live satellite and "
            f"reanalysis data."
        )

        # --- Paragraph 2: Signal breakdown ---
        p2_parts = []
        if prediction_data:
            p2_parts.append(
                f"Primary signal: **GloFAS v4** river discharge classifies this location "
                f"as **{glofas_level}** (T{tier}: {source}), the same operational dataset "
                f"used by European flood early-warning services. "
                f"Discharge anomaly: **{anomaly_sigma:+.1f}σ** vs seasonal mean."
            )
            compound_parts = []
            if precip_7d > 0:
                anom_label = f"{precip_delta:+.0%} anomaly" if precip_delta != 0 else "near-normal"
                compound_parts.append(f"**{precip_7d:.0f}mm** 7-day ERA5 rainfall ({anom_label})")
            if soil_sat > 0:
                compound_parts.append(f"**{soil_sat:.0%}** soil saturation (ERA5/ECMWF IFS)")
            if compound_parts:
                p2_parts.append(f"Compound signals: {', '.join(compound_parts)}.")
            if precip_delta > 0.04:
                p2_parts.append(f"Elevated precipitation adds **+{precip_delta:.0%}** to the base flood probability.")
            elif precip_delta < -0.04:
                p2_parts.append(f"Below-normal precipitation reduces the base probability by **{abs(precip_delta):.0%}**.")
            if soil_delta > 0.04:
                p2_parts.append(f"High soil saturation adds a further **+{soil_delta:.0%}** compound uplift.")

        p2 = " ".join(p2_parts) if p2_parts else (
            "Live signal data will be available once the tiered assessment completes."
        )

        # --- Highlights ---
        highlights = [
            f"Risk Level: **{risk_level}** — {pred_prob:.0%} probability, {pred_conf:.0%} confidence",
        ]
        if prediction_data:
            highlights.append(f"GloFAS v4 river classification: {glofas_level} (T{tier})")
            if anomaly_sigma != 0:
                highlights.append(f"Discharge anomaly: {anomaly_sigma:+.1f}σ vs seasonal mean")
            if precip_7d > 0:
                highlights.append(f"7-day ERA5 rainfall: {precip_7d:.0f}mm")
            if soil_sat > 0:
                highlights.append(f"Soil saturation: {soil_sat:.0%} (ERA5/ECMWF IFS)")

        # --- Risk trend ---
        risk_trend_map = {"CRITICAL": "critical", "HIGH": "escalating", "MEDIUM": "stable", "LOW": "improving"}
        risk_trend = risk_trend_map.get(risk_level, "stable")

        narrative = f"{p1}\n\n{p2}"

        return {
            "narrative": narrative,
            "highlights": highlights,
            "risk_trend": risk_trend,
            "generated_at": datetime.utcnow().isoformat(),
            "engine": "template",
        }
