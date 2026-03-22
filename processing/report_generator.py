"""
Phase 5A: Regulatory Report Generator.

Generates structured reports in multiple formats:
  - TCFD (Task Force on Climate-related Financial Disclosures)
  - Sendai Framework compliance
  - Executive Summary PDF (reportlab)
"""
import io
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.reports")

# ── Risk colour palette (RGB 0-1) ─────────────────────────────────────────────
_RISK_COLORS = {
    "CRITICAL": (0.94, 0.27, 0.27),
    "HIGH":     (0.97, 0.62, 0.12),
    "MEDIUM":   (0.92, 0.78, 0.08),
    "LOW":      (0.13, 0.77, 0.37),
    "UNKNOWN":  (0.55, 0.55, 0.55),
}
_NAVY   = (0.05, 0.09, 0.18)   # page header / section headers
_DARK   = (0.13, 0.16, 0.22)   # body text dark
_MID    = (0.38, 0.42, 0.50)   # secondary text
_LIGHT  = (0.93, 0.95, 0.97)   # alternate row
_ACCENT = (0.00, 0.60, 0.80)   # cyan accent line


def _rgb(t):
    from reportlab.lib.colors import Color
    return Color(*t)


class ReportGenerator:
    """Generates structured regulatory and executive reports."""

    def __init__(self):
        logger.info("ReportGenerator initialized")

    # ──────────────────────────────────────────────────────────────────────────
    # Public PDF entry-point
    # ──────────────────────────────────────────────────────────────────────────

    def generate_pdf(
        self,
        region_name: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        risk_data: Optional[dict] = None,
        prediction_data: Optional[dict] = None,
        explanation_data: Optional[dict] = None,
        forecast_data: Optional[dict] = None,
        compound_data: Optional[dict] = None,
        financial_data: Optional[dict] = None,
        nlg_data: Optional[dict] = None,
        discharge_data: Optional[dict] = None,
    ) -> bytes:
        """
        Build a professional PDF report and return raw bytes.
        Caller should stream with Content-Type: application/pdf.
        """
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether,
        )
        from reportlab.lib.colors import HexColor, white, black

        buf = io.BytesIO()
        PAGE_W, PAGE_H = A4
        MARGIN = 20 * mm

        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=28 * mm, bottomMargin=20 * mm,
            title=f"COSMEON Flood Risk Report — {region_name}",
            author="COSMEON Flood Intelligence Platform",
        )

        styles = getSampleStyleSheet()
        W = PAGE_W - 2 * MARGIN  # usable width

        # ── Custom styles ──────────────────────────────────────────────────────
        def S(name, **kw):
            base = kw.pop("parent", "Normal")
            try:
                return ParagraphStyle(name, parent=styles[base], **kw)
            except KeyError:
                return ParagraphStyle(name, **kw)

        s_title    = S("CoTitle",    fontSize=22, leading=28, textColor=_rgb(_NAVY),
                        fontName="Helvetica-Bold", spaceAfter=2)
        s_subtitle = S("CoSub",      fontSize=10, leading=14, textColor=_rgb(_MID),
                        fontName="Helvetica", spaceAfter=8)
        s_h2       = S("CoH2",       fontSize=13, leading=18, textColor=_rgb(_NAVY),
                        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
        s_h3       = S("CoH3",       fontSize=10, leading=14, textColor=_rgb(_ACCENT),
                        fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
        s_body     = S("CoBody",     fontSize=9,  leading=13, textColor=_rgb(_DARK),
                        fontName="Helvetica", spaceAfter=4)
        s_small    = S("CoSmall",    fontSize=8,  leading=11, textColor=_rgb(_MID),
                        fontName="Helvetica", spaceAfter=2)
        s_center   = S("CoCenter",   fontSize=9,  leading=13, textColor=_rgb(_DARK),
                        fontName="Helvetica", alignment=TA_CENTER)
        s_label    = S("CoLabel",    fontSize=7.5, leading=10, textColor=_rgb(_MID),
                        fontName="Helvetica-Bold", spaceAfter=1)
        s_value    = S("CoValue",    fontSize=11, leading=15, textColor=_rgb(_DARK),
                        fontName="Helvetica-Bold")
        s_risk_lbl = S("CoRisk",     fontSize=10, leading=14,
                        fontName="Helvetica-Bold", alignment=TA_CENTER)

        # ── Helpers ────────────────────────────────────────────────────────────
        def accent_rule():
            return HRFlowable(width="100%", thickness=2, color=_rgb(_ACCENT), spaceAfter=6)

        def thin_rule():
            return HRFlowable(width="100%", thickness=0.5, color=_rgb(_LIGHT), spaceAfter=4)

        def risk_badge_table(level: str, prob: Optional[float] = None):
            """Coloured risk level badge as a single-cell table."""
            col = _RISK_COLORS.get(level, _RISK_COLORS["UNKNOWN"])
            label = level
            if prob is not None:
                label = f"{level}  {prob:.0%}"
            cell = Paragraph(label, s_risk_lbl)
            t = Table([[cell]], colWidths=[55 * mm], rowHeights=[10 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _rgb(col)),
                ("TEXTCOLOR",     (0, 0), (-1, -1), white),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ROUNDEDCORNERS", [3]),
            ]))
            return t

        def kv_table(rows, col_w=None):
            """Two-column label/value table with alternating row shading."""
            if col_w is None:
                col_w = [W * 0.38, W * 0.62]
            data = []
            for label, value in rows:
                data.append([
                    Paragraph(str(label), s_label),
                    Paragraph(str(value), s_body),
                ])
            t = Table(data, colWidths=col_w)
            ts = [
                ("VALIGN",      (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",(0, 0), (-1, -1), 6),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0,0), (-1, -1), 4),
                ("LINEBELOW",   (0, 0), (-1, -2), 0.3, _rgb(_LIGHT)),
            ]
            for i in range(0, len(data), 2):
                ts.append(("BACKGROUND", (0, i), (-1, i), _rgb(_LIGHT)))
            t.setStyle(TableStyle(ts))
            return t

        def section_header(title: str):
            return [Spacer(1, 4), Paragraph(title, s_h2), accent_rule()]

        # ── Assemble story ─────────────────────────────────────────────────────
        story = []
        now_str = datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
        risk    = risk_data or {}
        pred    = prediction_data or {}
        expl    = explanation_data or {}
        fcast   = forecast_data or {}
        comp    = compound_data or {}
        fin     = financial_data or {}
        nlg     = nlg_data or {}
        dis     = discharge_data or {}

        ml      = expl.get("ml_prediction", {})
        risk_level = (
            ml.get("risk_level")
            or pred.get("predicted_risk_level")
            or risk.get("risk_level", "UNKNOWN")
        )
        flood_prob = (
            ml.get("probability")
            or pred.get("flood_probability")
        )
        confidence = (
            ml.get("confidence")
            or pred.get("confidence")
            or risk.get("confidence_score")
        )

        # ── Cover block ────────────────────────────────────────────────────────
        story.append(Paragraph("COSMEON", s_title))
        story.append(Paragraph("Flood Intelligence Platform", s_subtitle))
        story.append(accent_rule())

        # Title row: region name left, risk badge right
        badge = risk_badge_table(
            risk_level,
            flood_prob if flood_prob is not None else None,
        )
        title_row = Table(
            [[Paragraph(f"Flood Risk Report<br/><font size=11 color='#606878'>{region_name}</font>",
                        S("CoTitleReg", fontSize=16, leading=22,
                          textColor=_rgb(_NAVY), fontName="Helvetica-Bold")),
              badge]],
            colWidths=[W - 60 * mm, 60 * mm],
        )
        title_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",  (1, 0), (1, 0),  "RIGHT"),
        ]))
        story.append(title_row)
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Generated: {now_str}", s_small))

        if lat is not None and lon is not None:
            story.append(Paragraph(f"Coordinates: {lat:.4f}°N, {lon:.4f}°E", s_small))

        story.append(thin_rule())

        # ── Section 1: Current Assessment ─────────────────────────────────────
        story += section_header("1. Current Risk Assessment")

        rows = [
            ("Risk Level",      risk_level),
            ("Flood Probability", f"{flood_prob:.1%}" if flood_prob is not None else "—"),
            ("Confidence",      f"{confidence:.1%}" if confidence is not None else "—"),
        ]
        if risk.get("flood_area_km2") is not None:
            rows.append(("Flood Area", f"{risk['flood_area_km2']:.1f} km²"))
        if risk.get("flood_percentage") is not None:
            rows.append(("Flood Coverage", f"{risk['flood_percentage'] * 100:.2f}%"))
        if risk.get("timestamp"):
            rows.append(("Last Satellite Assessment", str(risk["timestamp"])[:19]))

        story.append(kv_table(rows))

        # NLG narrative
        if nlg.get("narrative"):
            story += section_header("2. AI-Generated Assessment")
            story.append(Paragraph(nlg["narrative"], s_body))
            if nlg.get("highlights"):
                for h in nlg["highlights"]:
                    story.append(Paragraph(f"• {h}", s_body))

        # ── Section: ML Prediction Drivers ────────────────────────────────────
        if ml.get("top_drivers"):
            story += section_header("3. ML Prediction — Top Drivers")
            story.append(Paragraph(
                "The following signals had the highest influence on the current prediction "
                "(TieredFloodPredictor · GloFAS v4 + ERA5 reanalysis):",
                s_body,
            ))

            driver_data = [
                [Paragraph("Signal", s_label),
                 Paragraph("Value", s_label),
                 Paragraph("Influence", s_label),
                 Paragraph("Importance", s_label)],
            ]
            for d in ml["top_drivers"][:8]:
                feat = d.get("feature", "").replace("_", " ").title()
                val  = d.get("value")
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                inf  = d.get("influence", "neutral").upper()
                imp  = d.get("importance", 0)
                inf_color = "#ef4444" if inf == "INCREASES RISK" else "#22c55e" if inf == "DECREASES RISK" else "#9ca3af"
                driver_data.append([
                    Paragraph(feat, s_body),
                    Paragraph(val_str, s_center),
                    Paragraph(f'<font color="{inf_color}">{inf}</font>', s_center),
                    Paragraph(f"{imp:.3f}", s_center),
                ])

            col_w = [W * 0.38, W * 0.18, W * 0.30, W * 0.14]
            dt = Table(driver_data, colWidths=col_w)
            ts = [
                ("BACKGROUND",   (0, 0), (-1, 0),  _rgb(_NAVY)),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  white),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0),  8),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [_rgb(_LIGHT), white]),
                ("GRID",         (0, 0), (-1, -1), 0.3, _rgb(_LIGHT)),
                ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]
            dt.setStyle(TableStyle(ts))
            story.append(dt)

            if ml.get("explanation"):
                story.append(Spacer(1, 4))
                story.append(Paragraph(ml["explanation"], s_small))

        # ── Section: GloFAS River Discharge ───────────────────────────────────
        if dis:
            story += section_header("4. GloFAS v4 River Discharge")
            rows = []
            if dis.get("current_discharge_m3s") is not None:
                rows.append(("Current Discharge", f"{dis['current_discharge_m3s']:.1f} m³/s"))
            if dis.get("mean_discharge_m3s") is not None:
                rows.append(("Mean Discharge (30-yr)", f"{dis['mean_discharge_m3s']:.1f} m³/s"))
            if dis.get("discharge_anomaly_sigma") is not None:
                rows.append(("Anomaly (σ)", f"{dis['discharge_anomaly_sigma']:+.2f}σ"))
            if dis.get("flood_risk_level"):
                rows.append(("GloFAS Risk Level", dis["flood_risk_level"]))
            if rows:
                story.append(kv_table(rows))

        # ── Section: 6-Month Forecast ─────────────────────────────────────────
        if fcast.get("monthly_forecast"):
            story += section_header("5. Predictive Forecast (6-Month)")
            summary = fcast.get("summary", {})
            if summary:
                story.append(Paragraph(
                    f"Peak risk month: <b>{summary.get('peak_risk_month', '—')}</b>  ·  "
                    f"Peak probability: <b>{summary.get('peak_probability', 0):.0%}</b>  ·  "
                    f"Trend: <b>{summary.get('overall_trend', '—').upper()}</b>",
                    s_body,
                ))
                story.append(Spacer(1, 4))

            fcast_header = [
                [Paragraph(h, s_label) for h in
                 ["Month", "Risk Level", "Prob.", "Precip. (mm)", "Confidence"]],
            ]
            for m in fcast["monthly_forecast"][:6]:
                lvl = m.get("risk_level", "—")
                col = _RISK_COLORS.get(lvl, _RISK_COLORS["UNKNOWN"])
                fcast_header.append([
                    Paragraph(m.get("month_name", m.get("month", ""))[:7], s_body),
                    Paragraph(f'<font color="#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"><b>{lvl}</b></font>', s_body),
                    Paragraph(f"{m.get('risk_probability', 0):.0%}", s_center),
                    Paragraph(f"{m.get('precipitation_forecast_mm', 0):.0f}", s_center),
                    Paragraph(f"{m.get('confidence_lower', 0):.0%}–{m.get('confidence_upper', 0):.0%}", s_center),
                ])

            col_w = [W * 0.20, W * 0.22, W * 0.15, W * 0.22, W * 0.21]
            ft = Table(fcast_header, colWidths=col_w)
            ft.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  _rgb(_NAVY)),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0),  8),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rgb(_LIGHT), white]),
                ("GRID",          (0, 0), (-1, -1), 0.3, _rgb(_LIGHT)),
                ("LEFTPADDING",   (0, 0), (-1, -1), 5),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(ft)

        # ── Section: Compound Risk ─────────────────────────────────────────────
        if comp:
            story += section_header("6. Compound Risk (INFORM Index)")
            score = comp.get("compound_score", 0)
            level = comp.get("compound_level", "—")
            story.append(Paragraph(
                f"INFORM Composite Score: <b>{score:.3f}</b>  ·  Level: <b>{level}</b>  ·  "
                f"Dominant Hazard: <b>{comp.get('dominant_hazard', '—').replace('_', ' ').title()}</b>  ·  "
                f"Cascading Amplification: <b>{comp.get('cascading_amplification', 1):.2f}×</b>",
                s_body,
            ))

            if comp.get("hazard_layers"):
                story.append(Spacer(1, 4))
                hl_header = [[Paragraph(h, s_label) for h in ["Hazard Layer", "Severity", "Status"]]]
                for h in comp["hazard_layers"]:
                    name = h.get("name", "").replace("_", " ").title()
                    sev  = h.get("severity", 0)
                    sta  = h.get("status", "normal").upper()
                    sta_col = "#ef4444" if sta == "ACTIVE" else "#f97316" if sta == "WARNING" else "#22c55e"
                    hl_header.append([
                        Paragraph(name, s_body),
                        Paragraph(f"{sev:.2f}", s_center),
                        Paragraph(f'<font color="{sta_col}"><b>{sta}</b></font>', s_center),
                    ])
                hl_t = Table(hl_header, colWidths=[W * 0.55, W * 0.22, W * 0.23])
                hl_t.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  _rgb(_NAVY)),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, 0),  8),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rgb(_LIGHT), white]),
                    ("GRID",          (0, 0), (-1, -1), 0.3, _rgb(_LIGHT)),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story.append(hl_t)

            if comp.get("recommendations"):
                story.append(Spacer(1, 6))
                story.append(Paragraph("Recommendations", s_h3))
                for rec in comp["recommendations"]:
                    story.append(Paragraph(f"• {rec}", s_body))

        # ── Section: Financial Impact ──────────────────────────────────────────
        if fin:
            story += section_header("7. Financial Impact Estimate")
            rows = []
            if fin.get("total_impact_usd"):
                rows.append(("Total Economic Exposure", f"USD {fin['total_impact_usd']:,.0f}"))
            if fin.get("affected_population"):
                rows.append(("Affected Population", f"{fin['affected_population']:,}"))
            if fin.get("insurance_exposure_usd"):
                rows.append(("Insurance Exposure", f"USD {fin['insurance_exposure_usd']:,.0f}"))
            if fin.get("gdp_at_risk_pct"):
                rows.append(("GDP at Risk", f"{fin['gdp_at_risk_pct']:.2%}"))
            if rows:
                story.append(kv_table(rows))

        # ── Footer: data sources ───────────────────────────────────────────────
        story.append(Spacer(1, 10))
        story.append(thin_rule())
        story.append(Paragraph(
            "Data sources: GloFAS v4 (Copernicus Emergency Management Service) · "
            "ERA5 Reanalysis (ECMWF) · Open-Meteo Historical Climate API · "
            "World Bank Population & GDP · FAO-56 Penman-Monteith ET₀",
            s_small,
        ))
        story.append(Paragraph(
            "Methodology: TieredFloodPredictor (GloFAS v4 T1→T4 fallback) · "
            "INFORM Risk Index (EU JRC) · TCFD Disclosure Framework",
            s_small,
        ))
        story.append(Paragraph(
            f"COSMEON Flood Intelligence Platform  ·  {now_str}  ·  For authorised use only.",
            s_small,
        ))

        # ── Page header/footer callback ────────────────────────────────────────
        def on_page(canvas, doc):
            from reportlab.lib.colors import HexColor
            canvas.saveState()
            # Header bar
            canvas.setFillColor(_rgb(_NAVY))
            canvas.rect(0, PAGE_H - 16 * mm, PAGE_W, 16 * mm, fill=1, stroke=0)
            canvas.setFillColor(white)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawString(MARGIN, PAGE_H - 10 * mm, "COSMEON  ·  Flood Intelligence Platform")
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm, f"{region_name}  ·  {now_str}")
            # Footer
            canvas.setFillColor(_rgb(_MID))
            canvas.setFont("Helvetica", 7.5)
            canvas.drawCentredString(PAGE_W / 2, 12 * mm, f"Page {doc.page}  ·  Confidential — COSMEON")
            canvas.restoreState()

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return buf.getvalue()

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy JSON report methods (kept for API compatibility)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_tcfd_report(self, region_name, risk_data=None, forecast_data=None,
                              financial_data=None, compound_risk=None, fusion_data=None):
        now = datetime.utcnow().isoformat()
        risk = risk_data or {}
        forecast = forecast_data or {}
        financial = financial_data or {}
        compound = compound_risk or {}
        fusion = fusion_data or {}
        return {
            "report_type": "TCFD", "report_version": "1.0",
            "generated_at": now, "region": region_name,
            "sections": {
                "governance": {
                    "title": "Governance",
                    "content": (
                        f"Flood risk monitoring for {region_name} using Cosmeon's "
                        "satellite-based analysis platform."
                    ),
                    "data_sources": fusion.get("sensors_fused", ["sentinel-2"]),
                },
                "strategy": {
                    "title": "Strategy",
                    "current_risk_level": risk.get("risk_level", "UNKNOWN"),
                    "flood_coverage_pct": risk.get("flood_percentage", 0),
                    "forward_looking": {
                        "forecast_horizon": forecast.get("horizon_months", 0),
                        "peak_risk_month": forecast.get("summary", {}).get("peak_risk_month", "N/A"),
                        "trend": forecast.get("summary", {}).get("overall_trend", "unknown"),
                    },
                },
                "risk_management": {
                    "title": "Risk Management",
                    "compound_risk_score": compound.get("compound_score", 0),
                    "dominant_hazard": compound.get("dominant_hazard", "flood"),
                    "recommendations": compound.get("recommendations", []),
                },
                "metrics_and_targets": {
                    "title": "Metrics and Targets",
                    "total_financial_exposure": financial.get("total_impact_usd", 0),
                    "affected_population": financial.get("affected_population", 0),
                },
            },
        }

    def generate_sendai_report(self, region_name, risk_data=None,
                                financial_data=None, asset_data=None):
        risk = risk_data or {}
        financial = financial_data or {}
        assets = asset_data or {}
        return {
            "report_type": "Sendai Framework", "report_version": "1.0",
            "generated_at": datetime.utcnow().isoformat(), "region": region_name,
            "global_targets": {
                "target_a": {"name": "Reduce Disaster Mortality",
                             "value": financial.get("affected_population", 0), "status": "monitoring"},
                "target_c": {"name": "Reduce Economic Loss",
                             "value": financial.get("total_impact_usd", 0), "status": "monitoring"},
                "target_d": {"name": "Reduce Damage to Critical Infrastructure",
                             "value": assets.get("critical_assets", 0), "status": "monitoring"},
            },
            "risk_assessment": {
                "risk_level": risk.get("risk_level", "UNKNOWN"),
                "confidence": risk.get("confidence_score", 0),
                "flood_area_km2": risk.get("flood_area_km2", 0),
            },
        }

    def generate_executive_report(self, region_name, risk_data=None, nlg_summary=None,
                                   forecast_data=None, financial_data=None):
        risk = risk_data or {}
        nlg = nlg_summary or {}
        forecast = forecast_data or {}
        financial = financial_data or {}
        return {
            "report_type": "Executive Summary",
            "generated_at": datetime.utcnow().isoformat(),
            "region": region_name,
            "summary": {
                "narrative": nlg.get("narrative", f"Flood risk assessment for {region_name}."),
                "risk_level": risk.get("risk_level", "UNKNOWN"),
                "key_metrics": {
                    "flood_coverage": risk.get("flood_percentage", 0),
                    "confidence": risk.get("confidence_score", 0),
                    "financial_exposure": financial.get("total_impact_usd", 0),
                    "forecast_trend": forecast.get("summary", {}).get("overall_trend", "unknown"),
                },
                "highlights": nlg.get("highlights", []),
            },
        }

    def list_report_types(self):
        return [
            {"id": "tcfd",      "name": "TCFD Climate Disclosure",
             "description": "Task Force on Climate-related Financial Disclosures"},
            {"id": "sendai",    "name": "Sendai Framework",
             "description": "UN Sendai Framework for Disaster Risk Reduction"},
            {"id": "executive", "name": "Executive Summary",
             "description": "Concise executive overview with key metrics"},
        ]
