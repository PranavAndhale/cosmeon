"""
COSMEON Periodic Scheduler — auto-runs flood risk analysis on all regions.

Uses asyncio — runs inside the existing uvicorn event loop, no extra deps.
Default interval: 6 hours. Configurable via POST /api/scheduler/configure.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("cosmeon.scheduler")


class PeriodicScheduler:
    """
    Background asyncio task that periodically runs live analysis on all regions.

    Usage:
        scheduler = PeriodicScheduler(interval_hours=6)
        # Call scheduler.start(analysis_engine, db) on app startup
    """

    def __init__(self, interval_hours: int = 6):
        self._interval_hours = max(1, interval_hours)
        self._enabled = True
        self._last_run: Optional[str] = None
        self._next_run: Optional[str] = None
        self._runs_completed = 0
        self._task: Optional[asyncio.Task] = None
        self._analysis_engine = None
        self._db = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, analysis_engine, db) -> None:
        """Start the background scheduler loop. Call once on startup."""
        self._analysis_engine = analysis_engine
        self._db = db
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "PeriodicScheduler started — interval=%dh, first run in %dh",
            self._interval_hours, self._interval_hours,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._enabled = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("PeriodicScheduler stopped")

    def configure(self, interval_hours: Optional[int] = None, enabled: Optional[bool] = None) -> None:
        """Reconfigure interval or enable/disable. Restarts the loop."""
        if interval_hours is not None:
            self._interval_hours = max(1, min(interval_hours, 168))  # 1h – 7d
        if enabled is not None:
            self._enabled = enabled

        # Cancel current task and restart if enabled
        if self._task and not self._task.done():
            self._task.cancel()
        if self._enabled and self._analysis_engine:
            self._task = asyncio.create_task(self._loop())
        logger.info(
            "PeriodicScheduler configured — interval=%dh, enabled=%s",
            self._interval_hours, self._enabled,
        )

    def trigger_now(self) -> None:
        """Trigger an immediate analysis run (cancels current sleep, reruns)."""
        if self._task and not self._task.done():
            self._task.cancel()
        if self._analysis_engine:
            self._task = asyncio.create_task(self._run_then_loop())
        logger.info("PeriodicScheduler: manual trigger requested")

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "interval_hours": self._interval_hours,
            "last_run": self._last_run,
            "next_run": self._next_run,
            "runs_completed": self._runs_completed,
            "task_active": self._task is not None and not self._task.done(),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main periodic loop: sleep → analyze → repeat."""
        while self._enabled:
            sleep_secs = self._interval_hours * 3600
            self._next_run = (datetime.utcnow() + timedelta(seconds=sleep_secs)).isoformat()
            try:
                await asyncio.sleep(sleep_secs)
            except asyncio.CancelledError:
                break
            if not self._enabled:
                break
            await self._do_analysis()

    async def _run_then_loop(self) -> None:
        """Run analysis immediately, then restart normal loop."""
        await self._do_analysis()
        if self._enabled:
            self._task = asyncio.create_task(self._loop())

    async def _do_analysis(self) -> None:
        """Run live analysis on all DB regions."""
        try:
            logger.info("PeriodicScheduler: starting auto-analysis of all regions")
            regions = self._db.get_all_regions()
            if not regions:
                logger.info("PeriodicScheduler: no regions found, skipping")
                return
            regions_data = [{"id": r.id, "name": r.name, "bbox": r.bbox} for r in regions]
            results = self._analysis_engine.analyze_all_regions(regions_data)
            self._runs_completed += 1
            self._last_run = datetime.utcnow().isoformat()
            alerts = sum(1 for r in results if r.alert_triggered)
            logger.info(
                "PeriodicScheduler: run #%d complete — %d regions, %d alerts",
                self._runs_completed, len(results), alerts,
            )
        except Exception as e:
            logger.error("PeriodicScheduler: analysis failed — %s", e, exc_info=True)
