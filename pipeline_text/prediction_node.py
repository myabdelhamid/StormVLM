"""
StormVLM — Natural Language Prediction Node (Pk-Text)
======================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Ablation Study: Structured JSON vs. Natural Language Text

This node is the TEXT COUNTERPART of PredictionNode. It receives IDENTICAL
inputs (perception results with per-view detections, filtered objects from
categorization) and applies the SAME level of prompt engineering — but
requires the VLM to respond in structured natural language instead of JSON.

Differences from prediction_node.py (JSON version):
  • Output format: analytical narrative with headers (no JSON)
  • No json.loads(), no regex JSON extraction
  • Post-processing: physics-based rules applied via prompt instructions
    (the JSON version patches fields in Python after inference)
  • Object filtering: done in Python identically, but fed as a text list

Similarities (controlled variables):
  ✓ Same perception-to-prediction object filtering logic
  ✓ Same pre-computed motion hints (v_rel_x → approaching/diverging)
  ✓ Same physics rules (off-road=NO RISK, diverging=NO RISK, etc.)
  ✓ Same model, temperature, image resolution
"""

from __future__ import annotations

import glob
import logging
import os
import re
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image  # type: ignore[import]
from .perception_node import PerceptionResult, _MAX_IMAGE_SIZE  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.prediction_t")


@dataclass
class PredictionResultText:
    """Prediction result — natural language text."""
    frame_id: str
    prediction_text: str = ""

    def full_report(self) -> str:
        lines = [
            f"{'='*60}",
            f"PREDICTION REPORT (Text) — Frame {self.frame_id}",
            f"{'='*60}",
            self.prediction_text,
            f"\n{'='*60}",
        ]
        return "\n".join(lines)


class PredictionNodeText:
    """
    Natural-language prediction node (ablation counterpart of PredictionNode).

    Same filtering, same motion hints, same physics rules — but expressed
    as prompt instructions instead of Python post-processing.
    Output: analytical prose instead of JSON.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        temperature: float = 0.1,
        model=None,
        processor=None,
        max_words: int = 250,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self._model = model
        self._processor = processor
        self.max_words = max_words

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[PredictionNodeText] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[PredictionNodeText] Model loaded ✓")

    def predict_frame(
        self,
        perception_result: PerceptionResult,
        data_dir: str | Path,
        images: Optional[list[Image.Image]] = None,
    ) -> PredictionResultText:
        """
        Run prediction using perception outputs.
        Same filtering as JSON version, but text output.
        """
        self._ensure_model()
        data_dir = Path(data_dir)

        # Load front camera (same as JSON version)
        if images is None:
            matches = glob.glob(
                str(data_dir / f"{perception_result.frame_id}_camera0.*")
            )
            if matches:
                img = Image.open(matches[0]).convert("RGB")
            else:
                img = Image.new("RGB", (800, 600), (128, 128, 128))
        else:
            img = images[0]

        # ──────────────────────────────────────────────────────────────────
        # OBJECT FILTERING — IDENTICAL to JSON version
        # We extract categorized objects from perception, compute
        # motion hints, and build the same filtered_objects list.
        # ──────────────────────────────────────────────────────────────────
        filtered_objects = self._extract_filtered_objects(perception_result)

        prompt = self._build_prediction_prompt(
            filtered_objects, perception_result.ego_speed
        )

        temp_path = self._save_temp_image(img)
        log.info(
            f"[PredictionNodeText] Querying LVLM with "
            f"{len(filtered_objects)} valid objects..."
        )
        reasoning = self._query_model(temp_path, prompt)

        try:
            os.unlink(temp_path)
        except OSError:
            pass

        # Clean text output (no JSON parsing)
        reasoning = self._clean_text_output(reasoning, self.max_words)

        return PredictionResultText(
            frame_id=perception_result.frame_id,
            prediction_text=reasoning,
        )

    # ── Object filtering (IDENTICAL logic to PredictionNode) ─────────────
    @staticmethod
    def _extract_filtered_objects(
        perception_result: PerceptionResult,
    ) -> list[dict]:
        """
        Extract categorized objects from perception output.

        This mirrors PredictionNode's filtering logic exactly:
        - Parse perception reasoning (JSON or text) to find categories
        - Compute v_rel_x and motion hints
        - Skip 'Unknown' objects
        """
        filtered = []

        for view_name, vp in perception_result.views.items():
            # Try to extract categories from perception reasoning
            # (works with JSON perception output)
            categories = PredictionNodeText._extract_categories(vp.reasoning)

            class_counts: dict[str, int] = {}
            top_dets = sorted(
                vp.detections, key=lambda d: d.get("dist", float("inf"))
            )[:3]

            det_map: dict[str, dict] = {}
            for d in top_dets:
                cls_name = d.get("class", "object").lower()
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                obj_tag = f"<{cls_name}_{class_counts[cls_name]}>"
                det_map[obj_tag] = d

            for obj_id, d in det_map.items():
                # Get category from perception
                cat_info = categories.get(obj_id, {})
                category = cat_info.get("category", "Unknown")

                # For text perception output, try to infer category from text
                if not categories and vp.reasoning:
                    category = PredictionNodeText._infer_category_from_text(
                        obj_id, vp.reasoning
                    )

                if category == "Unknown":
                    continue

                speed = d.get("speed", 0.0)
                dist = d.get("dist", 0.0)
                speed_xyz = d.get("speed_x_y_z", [0.0, 0.0, 0.0])
                vx = speed_xyz[0]
                ego_speed = perception_result.ego_speed
                v_rel_x = vx - ego_speed

                # Motion hint computation (IDENTICAL to JSON version)
                if abs(vx) < 0.5 and abs(speed_xyz[1]) < 0.5:
                    motion = "Stationary"
                elif view_name == "Front":
                    motion = (
                        "Approaching (Closing in)"
                        if v_rel_x < 0
                        else "Moving away (Diverging)"
                    )
                elif view_name == "Back":
                    motion = (
                        "Approaching from behind (Closing in)"
                        if v_rel_x > 0
                        else "Moving away (Falling behind)"
                    )
                else:
                    motion = (
                        "Moving forwards relative to ego"
                        if v_rel_x > 0
                        else "Moving backwards relative to ego"
                    )

                location = cat_info.get("location_in_scene", "On Road")
                clean_id = obj_id.strip("<>")

                filtered.append({
                    "id": f"{view_name}_{clean_id}",
                    "category": category,
                    "location_in_scene": location,
                    "view": view_name,
                    "dist_m": round(dist, 1),
                    "speed_ms": round(speed, 1),
                    "v_rel_x": round(v_rel_x, 1),
                    "motion_hint": motion,
                })

        return filtered

    @staticmethod
    def _extract_categories(reasoning: str) -> dict[str, dict]:
        """Try to extract categories from perception reasoning (JSON or text)."""
        # Try JSON extraction (from JSON perception node)
        json_match = re.search(r'```json\s*(.*?)\s*```', reasoning, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("categorization", {})
            except (json.JSONDecodeError, AttributeError):
                pass

        # Try raw JSON
        raw_match = re.search(r'\{.*\}', reasoning, re.DOTALL)
        if raw_match:
            try:
                data = json.loads(raw_match.group(0))
                return data.get("categorization", {})
            except (json.JSONDecodeError, AttributeError):
                pass

        return {}

    @staticmethod
    def _infer_category_from_text(obj_id: str, text: str) -> str:
        """Infer object category from natural language perception text."""
        text_lower = text.lower()
        obj_id_lower = obj_id.lower()

        # Find the section talking about this object
        # Look for the object tag and the surrounding text
        idx = text_lower.find(obj_id_lower)
        if idx == -1:
            return "Unknown"

        # Get context around the object mention (200 chars after)
        context = text_lower[idx:idx + 200]

        if "confirmed" in context:
            return "Confirmed"
        elif "fogged" in context:
            return "Fogged"
        elif "ghost" in context:
            return "Ghost"
        elif "unknown" in context:
            return "Unknown"

        # Infer from visibility + radar mentions
        visible = "visible" in context and "not visible" not in context
        radar = "radar anchor" in context or "radar" in context

        if visible and radar:
            return "Confirmed"
        elif not visible and radar:
            return "Fogged"
        elif visible and not radar:
            return "Ghost"

        return "Unknown"

    # ══════════════════════════════════════════════════════════════════════
    # Prediction Prompt — TEXT VERSION
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _build_prediction_prompt(
        filtered_objects: list[dict],
        ego_speed: float,
    ) -> str:
        """
        Build a highly optimized NATURAL LANGUAGE prediction prompt.

        Same data as JSON version. Same physics rules — but encoded
        as reasoning instructions instead of Python post-processing.
        """
        prompt_parts = [
            "You are the StormVLM Prediction Engine. Your task is to predict "
            "the future motion and assess the collision risk for each detected "
            "object in a driving scene.",
            "",
            f"The ego vehicle is traveling at {ego_speed:.1f} m/s.",
            "",
            "## Detected Objects",
        ]

        if not filtered_objects:
            prompt_parts.append("No actionable objects detected in the scene.")
        else:
            for obj in filtered_objects:
                prompt_parts.append(
                    f"- **{obj['id']}**: category={obj['category']}, "
                    f"distance={obj['dist_m']}m, speed={obj['speed_ms']}m/s, "
                    f"relative_velocity={obj['v_rel_x']}m/s, "
                    f"motion={obj['motion_hint']}, "
                    f"location={obj['location_in_scene']}"
                )

        prompt_parts.extend([
            "",
            "## Your Task",
            "Analyze each object step by step using Chain-of-Thought reasoning. "
            "For each object, write a bullet point assessment that includes:",
            "1. **Object ID** and its category",
            "2. **Future motion prediction**: Will it approach, diverge, or stay stationary? "
            "Use the motion hint and relative velocity to determine this.",
            "3. **Kinematic reasoning**: Explain WHY this object is or isn't dangerous, "
            "referencing its distance, speed, and trajectory.",
            "4. **Risk level**: Assign exactly one of: NO RISK, LOW, MODERATE, or HIGH.",
            "",
            "## Mandatory Physics Rules (YOU MUST FOLLOW THESE)",
            "- If an object is DIVERGING or MOVING AWAY → risk is always NO RISK, "
            "regardless of distance. It cannot collide with the ego vehicle.",
            "- If an object is ON SIDEWALK or OFF ROAD → risk is always NO RISK. "
            "It is outside the driving path.",
            "- If a FOGGED or CONFIRMED object is within 15m ahead and STATIONARY "
            "→ risk is HIGH. This is an imminent collision threat.",
            "- If a FOGGED or CONFIRMED object is within 15m ahead and APPROACHING "
            "→ risk is at least MODERATE.",
            "- GHOST objects (visible but no radar) that are close and stationary "
            "→ MODERATE risk. They may be real obstacles.",
            "",
            "## Output Format",
            "Write your analysis as a structured report with Markdown headers and "
            "bullet points. Do NOT use JSON, code blocks, or dictionaries.",
            "End each object's assessment on its own bullet line.",
            "",
        ])
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

    # ── Model query ──────────────────────────────────────────────────────
    def _save_temp_image(self, img: Image.Image) -> str:
        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        resized.save(path)
        return path

    def _query_model(self, image_path: str, prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]
        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=800,
            temperature=self.temperature,
            verbose=False,
            resize_shape=_MAX_IMAGE_SIZE,
        )
        raw = (
            result.text.strip()
            if hasattr(result, "text")
            else str(result).strip()
        )
        return raw

    # ── Text cleanup ─────────────────────────────────────────────────────
    @staticmethod
    def _clean_text_output(text: str, max_words: int = 250) -> str:
        """
        Clean text output: strip code artifacts, deduplicate, enforce budget.
        NO json.loads, NO JSON extraction.
        """
        # Strip accidental code fences
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'```', '', text)

        # Deduplicate repeated lines
        lines = text.split('\n')
        seen: set[str] = set()
        unique: list[str] = []
        for line in lines:
            key = line.strip().lower()
            if not key:
                if unique and unique[-1].strip() == '':
                    continue
                unique.append('')
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(line)

        text = '\n'.join(unique).strip()

        # Enforce word budget
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
            if text and text[-1] not in '.!?':
                text += '.'

        return text if text else "No predictions available."
