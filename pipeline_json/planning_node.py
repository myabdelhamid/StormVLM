"""
PlanningNode — Final step of the DriveLM graph (Perception → Prediction → Planning).

Action selection is done **deterministically in Python** using the prediction
data.  The VLM is then queried only for a natural‑language reasoning
sentence so the output stays readable and DriveLM‑style.
"""

import json
import logging
import os
import re
import tempfile
from typing import Optional

log = logging.getLogger("stormvlm.planning")

# ── Action Space ──────────────────────────────────────────────────────────
ACTIONS = [
    "KEEP_SPEED",
    "DECELERATE",
    "STOP",
    "EMERGENCY_BRAKE",
    "STEER_TO_AVOID",
]

# Priority mapping  (higher = more urgent)
_RISK_PRIORITY = {"HIGH": 3, "MODERATE": 2, "LOW": 1}


class PlanningNode:
    """
    Deterministic Planning + VLM reasoning sentence.

    1. Parse the prediction JSON.
    2. Apply physics‑based rules in Python to select the safest action.
    3. Ask the VLM only for the one‑sentence reasoning explanation.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        max_words: int = 120,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature

        # Shared or lazy‑loaded model
        self._model = model
        self._processor = processor

    # ── Model management ──────────────────────────────────────────────────
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[PlanningNode] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]

        self._model, self._processor = load(self.model_id)
        log.info("[PlanningNode] Model loaded ✓")

    # ── Public entry point ────────────────────────────────────────────────
    def plan_action(self, prediction_json: str, ego_speed: float) -> dict:
        """Return a planning dict with critical_object_id, selected_action,
        and planning_reasoning."""

        objects = self._parse_prediction(prediction_json)

        # Step 1: deterministic action selection
        critical_obj, action = self._select_action(objects, ego_speed)

        # Step 2: VLM reasoning sentence
        reasoning = self._generate_reasoning(
            prediction_json, ego_speed, critical_obj, action
        )

        return {
            "critical_object_id": critical_obj.get("id", "None") if critical_obj else "None",
            "selected_action": action,
            "planning_reasoning": reasoning,
        }

    # ── Deterministic action selection ────────────────────────────────────
    @staticmethod
    def _parse_prediction(prediction_json: str) -> list[dict]:
        """Extract the list of predicted objects from the prediction JSON."""
        # Strip markdown wrappers
        clean = prediction_json.strip()
        clean = re.sub(r'^```json\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            log.warning("[PlanningNode] Failed to parse prediction JSON.")
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "scene_prediction" in data:
            return data["scene_prediction"]
        return []

    @staticmethod
    def _select_action(
        objects: list[dict], ego_speed: float
    ) -> tuple[Optional[dict], str]:
        """
        Pure‑Python rule engine.

        Key principles
        ──────────────
        • Only objects **ahead** (Front_) matter for braking/stopping decisions.
        • Objects behind (Back_) or to the side (Left_, Right_) never trigger
          braking — that would cause a rear‑end collision.
        • Ghost objects lack radar confirmation; treat them as low‑priority
          unless they are HIGH risk **and** in front.
        • Fogged/Confirmed objects in front with HIGH risk → EMERGENCY_BRAKE.
        """
        if not objects:
            if ego_speed < 0.01:
                return None, "ACCELERATE"
            return None, "KEEP_SPEED"

        # Separate objects by spatial position
        front_objects = [o for o in objects if o.get("id", "").startswith("Front_")]
        # (Back_, Left_, Right_ are informational only for braking decisions)

        # ── If no objects are ahead, the path is clear ────────────────────
        if not front_objects:
            if ego_speed < 0.01:
                return None, "ACCELERATE"
            return None, "KEEP_SPEED"

        # ── Evaluate worst threat ahead ───────────────────────────────────
        worst: Optional[dict] = None
        worst_priority = 0

        for obj in front_objects:
            risk = obj.get("risk_level", "LOW").upper()
            
            # ── Rule: Completely ignore safe off-road / sidewalk objects
            if risk == "NO RISK" or risk == "NONE":
                continue
                
            cat = obj.get("category", "Unknown")
            priority = _RISK_PRIORITY.get(risk, 0)

            # Ghost objects are slightly deprioritized vs Fogged/Confirmed
            # at equal risk level, but NOT if they are MODERATE or HIGH
            if cat == "Ghost" and priority <= 1:
                priority = max(1, priority - 1)

            if priority > worst_priority:
                worst_priority = priority
                worst = obj

        if worst is None:
            if ego_speed < 0.01:
                return None, "ACCELERATE"
            return None, "KEEP_SPEED"

        risk = worst.get("risk_level", "LOW").upper()
        cat = worst.get("category", "Unknown")
        motion = worst.get("future_motion", "").lower()

        # ── Rule: Already stopped → never decelerate/brake further ────────
        # If ego is stationary, either KEEP_SPEED (wait) or ACCELERATE
        if ego_speed < 0.01:
            # HIGH risk threat directly ahead → stay stopped (KEEP_SPEED)
            if risk in ("HIGH", "MODERATE"):
                return worst, "KEEP_SPEED"
            # Otherwise safe to go
            return worst, "ACCELERATE"

        # ── Decision tree ─────────────────────────────────────────────────
        # Objects moving away from us ahead are not a threat
        if "diverging" in motion or "falling behind" in motion or "moving away" in motion:
            return worst, "KEEP_SPEED"

        # HIGH risk + Fogged/Confirmed ahead → EMERGENCY_BRAKE
        if risk == "HIGH" and cat in ("Fogged", "Confirmed", "Ghost"):
            return worst, "EMERGENCY_BRAKE"

        # MODERATE risk ahead → DECELERATE (covers close Ghost objects too)
        if risk == "MODERATE":
            return worst, "DECELERATE"

        # LOW risk — generally safe
        if ego_speed > 5.0:
            return worst, "DECELERATE"  # Extra caution in fog at speed

        return worst, "KEEP_SPEED"

    # ── VLM reasoning sentence ────────────────────────────────────────────
    def _generate_reasoning(
        self,
        prediction_json: str,
        ego_speed: float,
        critical_obj: Optional[dict],
        action: str,
    ) -> str:
        """Ask the VLM for a single DriveLM‑style sentence explaining the
        chosen action.  Falls back to a template if the VLM is unavailable."""
        self._ensure_model()

        obj_summary = "no forward threats detected"
        if critical_obj:
            obj_id = critical_obj.get("id", "unknown")
            cat = critical_obj.get("category", "unknown")
            risk = critical_obj.get("risk_level", "LOW")
            speed = critical_obj.get("speed", "unknown")
            motion = critical_obj.get("future_motion", "unknown")
            obj_summary = (
                f"critical object {obj_id} (category={cat}, risk={risk}, "
                f"speed={speed}, motion={motion})"
            )

        prompt = (
            f"You are a driving safety system. The ego vehicle is moving at "
            f"{ego_speed:.1f} m/s.  The selected action is \"{action}\".  "
            f"The scene context: {obj_summary}.  "
            f"Write ONE sentence (max 30 words) justifying the action.  "
            f"Output ONLY the sentence, nothing else."
        )

        try:
            from mlx_vlm import generate  # type: ignore[import]
            from PIL import Image  # type: ignore[import]

            # mlx_vlm requires an image — use a tiny black placeholder
            img = Image.new("RGB", (64, 64), color="black")
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img.save(temp_path)

            result = generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=[temp_path],
                max_tokens=80,
                temperature=self.temperature,
                verbose=False,
            )
            raw = result.text.strip() if hasattr(result, "text") else str(result).strip()

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Clean: take only first sentence, strip quotes/markdown
            raw = raw.strip('`"\' \n')
            first_sentence = raw.split(".")[0] + "." if "." in raw else raw
            return first_sentence

        except Exception as e:
            log.warning(f"[PlanningNode] VLM reasoning failed ({e}); using template.")
            # Fallback template
            if action == "KEEP_SPEED":
                return f"Path ahead is clear ({obj_summary}); maintaining speed."
            elif action == "DECELERATE":
                return f"Moderate risk from {obj_summary}; reducing speed for safety."
            elif action == "STOP":
                return f"Hazard ahead from {obj_summary}; performing controlled stop."
            elif action == "EMERGENCY_BRAKE":
                return f"Imminent collision threat from {obj_summary}; emergency braking."
            else:
                return f"Action {action} selected due to {obj_summary}."
