"""
Phase 5B: MLOps Feedback Engine.

Implements a continuous learning loop where users can verify or reject
model detections. Feedback is stored and used to track model performance
and identify areas for retraining.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.feedback")


@dataclass
class FeedbackEntry:
    """A single user feedback data point."""
    id: int = 0
    region_id: Optional[int] = None
    lat: float = 0.0
    lon: float = 0.0
    detection_type: str = ""  # "flood", "deforestation", "heat_stress"
    model_prediction: str = ""  # What the model said
    user_verdict: str = ""  # "correct", "incorrect", "uncertain"
    user_label: str = ""  # What the user says it actually is
    notes: str = ""
    submitted_at: str = ""
    applied_to_training: bool = False

    def to_dict(self):
        return {
            "id": self.id, "region_id": self.region_id,
            "lat": self.lat, "lon": self.lon,
            "detection_type": self.detection_type,
            "model_prediction": self.model_prediction,
            "user_verdict": self.user_verdict,
            "user_label": self.user_label,
            "notes": self.notes,
            "submitted_at": self.submitted_at,
            "applied_to_training": self.applied_to_training,
        }


class FeedbackEngine:
    """Manages user feedback for continuous model improvement."""

    def __init__(self):
        self._feedback: list[FeedbackEntry] = []
        self._next_id = 1
        logger.info("FeedbackEngine initialized")

    def submit_feedback(
        self,
        detection_type: str,
        model_prediction: str,
        user_verdict: str,
        user_label: str = "",
        notes: str = "",
        region_id: int = None,
        lat: float = 0.0,
        lon: float = 0.0,
    ) -> dict:
        """Submit user feedback on a model detection."""
        entry = FeedbackEntry(
            id=self._next_id,
            region_id=region_id,
            lat=lat, lon=lon,
            detection_type=detection_type,
            model_prediction=model_prediction,
            user_verdict=user_verdict,
            user_label=user_label,
            notes=notes,
            submitted_at=datetime.utcnow().isoformat(),
        )
        self._feedback.append(entry)
        self._next_id += 1

        logger.info(
            "Feedback submitted: %s prediction '%s' → user says '%s' (%s)",
            detection_type, model_prediction, user_verdict, user_label or "no label"
        )
        return entry.to_dict()

    def get_feedback_stats(self) -> dict:
        """Get feedback statistics for model performance tracking."""
        total = len(self._feedback)
        if total == 0:
            return {"total": 0, "accuracy": 0, "by_type": {}, "recent": []}

        correct = sum(1 for f in self._feedback if f.user_verdict == "correct")
        incorrect = sum(1 for f in self._feedback if f.user_verdict == "incorrect")
        uncertain = sum(1 for f in self._feedback if f.user_verdict == "uncertain")

        # By detection type
        by_type = {}
        for f in self._feedback:
            t = f.detection_type
            if t not in by_type:
                by_type[t] = {"total": 0, "correct": 0, "incorrect": 0}
            by_type[t]["total"] += 1
            if f.user_verdict == "correct":
                by_type[t]["correct"] += 1
            elif f.user_verdict == "incorrect":
                by_type[t]["incorrect"] += 1

        # Compute per-type accuracy
        for t in by_type:
            reviewed = by_type[t]["correct"] + by_type[t]["incorrect"]
            by_type[t]["accuracy"] = round(by_type[t]["correct"] / max(reviewed, 1), 3)

        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "uncertain": uncertain,
            "accuracy": round(correct / max(correct + incorrect, 1), 3),
            "by_type": by_type,
            "pending_training": sum(1 for f in self._feedback if not f.applied_to_training),
            "recent": [f.to_dict() for f in self._feedback[-10:]],
        }

    def get_training_candidates(self) -> list[dict]:
        """Get feedback entries that haven't been applied to training yet."""
        candidates = [
            f for f in self._feedback
            if not f.applied_to_training and f.user_verdict in ("correct", "incorrect")
        ]
        return [c.to_dict() for c in candidates]

    def mark_applied(self, feedback_ids: list[int]):
        """Mark feedback entries as applied to training."""
        for f in self._feedback:
            if f.id in feedback_ids:
                f.applied_to_training = True
        logger.info("Marked %d feedback entries as applied to training", len(feedback_ids))

    def get_misclassifications(self) -> list[dict]:
        """Get common misclassification patterns."""
        incorrect = [f for f in self._feedback if f.user_verdict == "incorrect"]
        patterns = {}
        for f in incorrect:
            key = f"{f.model_prediction} → {f.user_label or 'unknown'}"
            if key not in patterns:
                patterns[key] = {"count": 0, "examples": []}
            patterns[key]["count"] += 1
            if len(patterns[key]["examples"]) < 3:
                patterns[key]["examples"].append({
                    "lat": f.lat, "lon": f.lon, "notes": f.notes
                })

        return [
            {"pattern": k, "count": v["count"], "examples": v["examples"]}
            for k, v in sorted(patterns.items(), key=lambda x: -x[1]["count"])
        ]
