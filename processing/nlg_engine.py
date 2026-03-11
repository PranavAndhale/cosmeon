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
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.nlg")

# In-memory cache for generated summaries (region_id -> {hash, text, generated_at})
_nlg_cache: dict[int, dict] = {}


class NLGEngine:
    """Generates natural language summaries from structured analysis data."""

    def __init__(self):
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("NLG Engine initialized with OpenAI GPT integration")
            except ImportError:
                logger.warning("openai package not installed — using template-based NLG")
        else:
            logger.info("NLG Engine initialized (template-based, no OPENAI_API_KEY set)")

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

        # Try GPT-based generation first
        if self.openai_client:
            result = self._generate_with_gpt(
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

    def _generate_with_gpt(
        self,
        region_name: str,
        risk_data: dict,
        prediction_data: Optional[dict],
        detection_data: Optional[dict],
        validation_data: Optional[dict],
        external_factors: Optional[dict],
    ) -> dict:
        """Generate narrative using OpenAI GPT."""
        try:
            data_block = json.dumps({
                "region": region_name,
                "current_risk": risk_data,
                "ml_prediction": prediction_data,
                "live_detection": detection_data,
                "glofas_validation": validation_data,
                "weather_factors": external_factors,
            }, indent=2, default=str)

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a climate intelligence analyst for the Cosmeon satellite "
                            "flood monitoring platform. Generate concise, professional executive "
                            "summaries. Use specific numbers. Output JSON with keys: "
                            '"narrative" (2-3 paragraphs), "highlights" (3-5 bullet points), '
                            '"risk_trend" (one of: escalating, stable, improving, critical).'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Generate an executive summary for this flood analysis data:\n\n{data_block}",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.3,
            )

            content = json.loads(response.choices[0].message.content)
            content["generated_at"] = datetime.utcnow().isoformat()
            content["engine"] = "gpt-4o-mini"
            return content

        except Exception as e:
            logger.warning("GPT generation failed (%s), falling back to templates", e)
            return self._generate_with_templates(
                region_name, risk_data, prediction_data,
                detection_data, validation_data, external_factors
            )

    def _generate_with_templates(
        self,
        region_name: str,
        risk_data: dict,
        prediction_data: Optional[dict],
        detection_data: Optional[dict],
        validation_data: Optional[dict],
        external_factors: Optional[dict],
    ) -> dict:
        """Generate narrative using template-based string interpolation."""
        risk_level = risk_data.get("risk_level", "UNKNOWN")
        flood_pct = risk_data.get("flood_percentage", 0)
        confidence = risk_data.get("confidence_score", 0)
        flood_area = risk_data.get("flood_area_km2", 0)
        total_area = risk_data.get("total_area_km2", 0)

        # Severity context
        severity_phrases = {
            "CRITICAL": ("demands immediate attention", "critical"),
            "HIGH": ("warrants close monitoring", "elevated"),
            "MEDIUM": ("requires ongoing observation", "moderate"),
            "LOW": ("presents minimal concern at this time", "low"),
        }
        phrase, adj = severity_phrases.get(risk_level, ("requires review", risk_level.lower()))

        # --- Paragraph 1: Current situation ---
        p1 = (
            f"The current flood risk assessment for **{region_name}** indicates a "
            f"**{risk_level}** threat level that {phrase}. Satellite analysis reveals "
            f"that approximately **{flood_pct:.1%}** of the monitored area "
            f"({flood_area:.1f} km² out of {total_area:.0f} km²) shows flood indicators, "
            f"with a model confidence of **{confidence:.0%}**."
        )

        # --- Paragraph 2: Prediction + Detection ---
        p2_parts = []
        if prediction_data:
            pred_risk = prediction_data.get("predicted_risk_level", "N/A")
            pred_prob = prediction_data.get("flood_probability", 0)
            p2_parts.append(
                f"The ML ensemble model predicts a **{pred_risk}** risk level with a "
                f"flood probability of **{pred_prob:.0%}**."
            )
        if detection_data:
            det_risk = detection_data.get("detected_risk_level", "N/A")
            rainfall = detection_data.get("rainfall_7d_mm", 0)
            p2_parts.append(
                f"Automated live detection confirms a **{det_risk}** risk, with "
                f"**{rainfall:.0f}mm** of rainfall recorded over the past 7 days."
            )
        if validation_data and isinstance(validation_data, dict):
            val = validation_data.get("validation", validation_data)
            agreement = val.get("agreement", False)
            score = val.get("agreement_score", 0)
            status = "aligned" if agreement else "divergent"
            p2_parts.append(
                f"GloFAS river discharge validation is **{status}** with our prediction "
                f"(agreement score: {score:.0%})."
            )

        p2 = " ".join(p2_parts) if p2_parts else (
            "Additional prediction and validation data will be available once analysis completes."
        )

        # --- Highlights ---
        highlights = [
            f"Risk Level: {risk_level} ({confidence:.0%} confidence)",
            f"Flood Coverage: {flood_pct:.1%} of monitored area ({flood_area:.1f} km²)",
        ]
        if prediction_data:
            highlights.append(
                f"ML Prediction: {prediction_data.get('predicted_risk_level', 'N/A')} "
                f"({prediction_data.get('flood_probability', 0):.0%} probability)"
            )
        if detection_data:
            highlights.append(
                f"7-Day Rainfall: {detection_data.get('rainfall_7d_mm', 0):.0f}mm"
            )
        if external_factors:
            highlights.append(
                f"Elevation: {external_factors.get('elevation_mean_m', 'N/A')}m avg"
            )

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
