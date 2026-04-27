"""
StormVLM — Natural Language Perception Node (PA-Text)
======================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Ablation Study: Structured JSON vs. Natural Language Text

This node is the TEXT COUNTERPART of AnchoredPerceptionNode. It receives
IDENTICAL inputs (GT detections, radar anchors, CLAHE-enhanced images) and
applies the SAME level of prompt engineering — but requires the VLM to
respond in structured natural language (Markdown-style prose) instead of JSON.

Differences from perception_node.py (JSON version):
  • Output format: analytical paragraph with headers/bullets (no JSON)
  • No json.loads(), no regex JSON extraction, no dictionary mapping
  • Post-processing: text deduplication only (no JSON field overrides)
  • Close-range visibility override: embedded as a prompt instruction
    (the JSON version patches it in Python after inference)

Similarities (controlled variables):
  ✓ Same YAML ground-truth detections
  ✓ Same radar anchor pre-matching (Python-side radar match)
  ✓ Same CLAHE fog enhancement
  ✓ Same top-3-closest object filtering
  ✓ Same model, temperature, image resolution
"""

from __future__ import annotations

import glob
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np  # type: ignore[import]
from PIL import Image  # type: ignore[import]

from pipeline_json.perception_node import (
    AnnotationLoader,
    PerceptionResult,
    ViewPerception,
    _MAX_IMAGE_SIZE,
    _VIEW_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.perception_t")


class PerceptionNodeText:
    """
    Natural-language perception node (ablation counterpart of AnchoredPerceptionNode).

    Same inputs, same pre-processing, same radar augmentation.
    Output: analytical prose per view instead of JSON.
    """

    _CAMERA_TO_VIEW: dict[str, str] = {
        "Camera 0 (Front)": "Front",
        "Camera 1 (Right)": "Right",
        "Camera 2 (Left)": "Left",
        "Camera 3 (Back)": "Back",
    }

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        max_words: int = 200,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature
        self._model = None
        self._processor = None

    # ── Model loading (identical to JSON version) ────────────────────────
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[PerceptionNodeText] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[PerceptionNodeText] Model loaded ✓")

    # ── Main entry point ─────────────────────────────────────────────────
    def perceive_frame(
        self,
        data_dir: str | Path,
        frame_id: str,
        images: Optional[list[Image.Image]] = None,
        radar_anchors: Optional[list[dict]] = None,
    ) -> PerceptionResult:
        """
        Full perception pipeline — identical flow to JSON version,
        but VLM outputs natural language instead of JSON.
        """
        self._ensure_model()
        data_dir = Path(data_dir)

        # 1. Load YAML annotations (same)
        yaml_path = data_dir / f"{frame_id}.yaml"
        annotations = AnnotationLoader.load(yaml_path)

        # 2. Load camera images (same)
        if images is None:
            images = self._load_cameras(data_dir, frame_id)

        # 3. Load radar anchors (same)
        if radar_anchors is None:
            from stormvlm_loader import GlobalRadarFilter
            rf = GlobalRadarFilter()
            radar_anchors = rf.process(str(data_dir), frame_id)

        # 4. Group by view (same)
        view_map = self._group_by_view(annotations, radar_anchors)

        # 5. Weather string (same logic)
        weather_val = annotations.get("weather_type", "fd").lower()
        if "cd" in weather_val:
            scenario_str = "clear day"
        elif "fd" in weather_val:
            scenario_str = "heavy fog"
        elif "fhrd" in weather_val:
            scenario_str = "heavy fog and rain"
        else:
            scenario_str = "heavy fog"

        result = PerceptionResult(
            frame_id=frame_id,
            ego_speed=annotations.get("ego_speed", 0.0),
            weather_type=weather_val,
        )

        view_to_cam = {"Front": 0, "Right": 1, "Left": 2, "Back": 3}

        for view_name in _VIEW_NAMES:
            cam_idx = view_to_cam[view_name]
            detections = view_map.get(view_name, {}).get("detections", [])
            anchors = view_map.get(view_name, {}).get("anchors", [])

            # Build NATURAL LANGUAGE prompt (same data, different output format)
            prompt = self._build_prompt(
                view_name, detections, anchors, scenario=scenario_str
            )
            log.info(
                f"[PerceptionNodeText] {view_name} view: "
                f"{len(detections)} detections, {len(anchors)} anchors "
                f"→ querying LVLM..."
            )

            # Save temp image WITH CLAHE (same as JSON version)
            temp_path = self._save_temp_image(images[cam_idx])
            reasoning = self._query_model(temp_path, prompt)

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # No JSON post-processing — text deduplication only
            reasoning = self._clean_text_output(reasoning, self.max_words)

            result.views[view_name] = ViewPerception(
                view_name=view_name,
                detections=detections,
                radar_anchors=anchors,
                reasoning=reasoning,
            )

        return result

    # ── Camera loading (identical) ───────────────────────────────────────
    @staticmethod
    def _load_cameras(data_dir: Path, frame_id: str) -> list[Image.Image]:
        images = []
        for i in range(4):
            matches = glob.glob(str(data_dir / f"{frame_id}_camera{i}.*"))
            if matches:
                images.append(Image.open(matches[0]).convert("RGB"))
            else:
                log.warning(f"Camera {i} not found for frame {frame_id}")
                images.append(Image.new("RGB", (800, 600), (128, 128, 128)))
        return images

    # ── Image preprocessing WITH CLAHE (identical to JSON version) ───────
    @staticmethod
    def _save_temp_image(img: Image.Image) -> str:
        import cv2

        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(resized)

        # CLAHE on L channel (enhance contrast in fog) — SAME as JSON version
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        enhanced = Image.fromarray(arr)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        enhanced.save(path)
        return path

    # ── Group by view (identical) ────────────────────────────────────────
    @staticmethod
    def _group_by_view(
        annotations: dict, radar_anchors: list[dict]
    ) -> dict[str, dict]:
        grouped: dict[str, dict] = {
            v: {"detections": [], "anchors": []}
            for v in _VIEW_NAMES
        }
        for obj in annotations.get("vehicles", []) + annotations.get("walkers", []):
            view = obj.get("view", "Front")
            if view in grouped:
                grouped[view]["detections"].append(obj)
        for anchor in radar_anchors:
            primary = PerceptionNodeText._CAMERA_TO_VIEW.get(
                anchor.get("primary_camera", ""), None
            )
            if primary and primary in grouped:
                grouped[primary]["anchors"].append(anchor)
        return grouped

    # ══════════════════════════════════════════════════════════════════════
    # Prompt — THIS IS THE KEY DIFFERENCE
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _build_prompt(
        view_name: str,
        detections: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
        scenario: str = "heavy fog",
    ) -> str:
        """
        Build a highly optimized NATURAL LANGUAGE prompt.

        Same data injection as the JSON version:
          • Top-3 closest detections with class, distance, speed
          • Pre-computed radar match boolean
          • Weather scenario

        But instead of requesting a JSON schema, we ask for an analytical
        safety report using Chain-of-Thought reasoning with Markdown headers
        and bullet points.
        """
        # ── DATA SECTION (identical data as JSON version) ────────────────
        prompt_parts = [
            f"Here is the sensor data for the {view_name.lower()} view:",
            "Ground Truth Objects:"
        ]

        class_counts: dict[str, int] = {}
        sorted_detections = sorted(
            detections, key=lambda d: d.get("dist", float("inf"))
        )
        top_detections = sorted_detections[:3]

        if top_detections:
            for d in top_detections:
                cls_name = d.get("class", "object").lower()
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                obj_tag = f"<{cls_name}_{class_counts[cls_name]}>"
                speed = d.get("speed", 0.0)
                dist = d.get("dist", 0.0)
                status = "stopped" if speed < 0.5 else f"moving at {speed:.1f}m/s"
                # Pre-compute radar match (SAME logic as JSON version)
                radar_match = "False"
                for a in anchors:
                    dist_delta = abs(a["dist_m"] - dist)
                    speed_delta = abs(a["max_speed_ms"] - speed)
                    if dist_delta < 20.0 and speed_delta < 10.0:
                        radar_match = "True"
                        break
                prompt_parts.append(
                    f"- {obj_tag} at {dist:.1f}m, {status} "
                    f"(Has Radar Anchor: {radar_match})"
                )
        else:
            prompt_parts.append("- None")

        # ── INSTRUCTION SECTION (text-specific) ─────────────────────────
        instructions = (
            f"\nQuestion: You are an expert autonomous driving safety auditor "
            f"performing a perception analysis. The current weather is {scenario}.\n"
            "\n"
            "Analyze the scene step by step using the image and the sensor data above. "
            "Write your response as a structured safety report using the following format:\n"
            "\n"
            "## Weather Assessment\n"
            "Write one sentence describing the current visibility conditions and how they "
            "affect the camera's ability to detect objects.\n"
            "\n"
            "## Object-by-Object Analysis\n"
            "For EACH Ground Truth Object listed above, write a bullet point with:\n"
            "- The object tag (e.g., <car_1>)\n"
            "- Whether it is VISIBLE or NOT VISIBLE in the camera image "
            "(look at the image carefully; objects closer than 15 meters are almost always "
            "visible even in heavy fog)\n"
            "- Whether it has a RADAR ANCHOR (use the 'Has Radar Anchor' tag provided)\n"
            "- Its location: On Road, On Sidewalk, or Off Road "
            "(determine this by examining where the object appears in the image)\n"
            "- Its safety category based on this logic:\n"
            "  * CONFIRMED = visible in camera AND has radar anchor\n"
            "  * FOGGED = not visible in camera BUT has radar anchor\n"
            "  * GHOST = visible in camera BUT no radar anchor\n"
            "  * UNKNOWN = not visible AND no radar anchor\n"
            "\n"
            "CRITICAL RULES:\n"
            "- You MUST analyze EVERY object listed in Ground Truth Objects. Do not skip any.\n"
            "- Do NOT output JSON, code blocks, or dictionaries. Write in plain English with "
            "Markdown headers and bullet points only.\n"
            "- Be precise and concise. Each bullet should be one line.\n"
        )

        prompt_parts.append(instructions)
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

    # ── LVLM query (same model call, no JSON extraction) ─────────────────
    def _query_model(self, image_path: str, prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]

        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=500,
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

    # ── Text cleanup (NO JSON parsing) ───────────────────────────────────
    @staticmethod
    def _clean_text_output(text: str, max_words: int = 200) -> str:
        """
        Clean VLM text output:
        1. Strip any accidental JSON/code block artifacts.
        2. Deduplicate repeated sentences.
        3. Enforce word budget.

        NO json.loads, NO regex JSON extraction, NO dictionary mapping.
        """
        # Strip any accidental code fences the VLM might still emit
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'```', '', text)

        # Deduplicate repeated lines/sentences
        lines = text.split('\n')
        seen: set[str] = set()
        unique_lines: list[str] = []
        for line in lines:
            key = line.strip().lower()
            if not key:
                # Preserve blank lines for formatting (but max 1 consecutive)
                if unique_lines and unique_lines[-1].strip() == '':
                    continue
                unique_lines.append('')
                continue
            if key in seen:
                continue
            seen.add(key)
            unique_lines.append(line)

        text = '\n'.join(unique_lines).strip()

        # Enforce word budget
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
            if text and text[-1] not in '.!?':
                text += '.'

        return text if text else "No observations available."
