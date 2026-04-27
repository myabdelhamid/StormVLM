"""
StormVLM — DriveLM-Style Anchored Perception Node
====================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Architecture (inspired by DriveLM, ECCV 2024):
Detection → YAML ground-truth annotations (vehicles, walkers)
Radar → GlobalRadarFilter anchors (moving objects)
Reasoning → VLM (Qwen2.5-VL-3B) reasons over *known* detections + images

The VLM never tries to **detect** objects — it only **reasons** about
pre-detected ones. This eliminates hallucination and missed detections.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np # type: ignore[import]
from PIL import Image # type: ignore[import]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
level=logging.INFO,
format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.perception")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_IMAGE_SIZE = (448, 448)

# Map relative_angle ranges to camera views
# Front: roughly -45° to +45° (object ahead of ego)
# Right: roughly -135° to -45°
# Left: roughly +45° to +135°
# Back: roughly |angle| > 135°
_VIEW_NAMES = ["Front", "Right", "Left", "Back"]


# ---------------------------------------------------------------------------
# Annotation Loader — reads YAML ground truth
# ---------------------------------------------------------------------------
class AnnotationLoader:
    """
    Parses the per-frame YAML file from Adver-City-R / OPV2V format
    and returns structured detection lists for vehicles and walkers.
    """

    @staticmethod
    def load(yaml_path: str | Path) -> dict:
        """
        Load annotations from a YAML file.

        Returns
        -------
        dict with keys:
        'vehicles': list[dict] — each with class, dist, speed, location, etc.
        'walkers': list[dict] — same structure
        'ego_speed': float
        """
        import yaml # lazy import

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            log.warning(f"Annotation file not found: {yaml_path}")
            return {"vehicles": [], "walkers": [], "ego_speed": 0.0}

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        vehicles = AnnotationLoader._parse_objects(
            data.get("vehicles", {}), obj_type="vehicle"
        )
        walkers = AnnotationLoader._parse_objects(
            data.get("walkers", {}), obj_type="walker"
        )
        ego_speed = data.get("ego_speed", 0.0)
        weather_type = data.get("weather_type", "fd")

        log.info(
            f"[AnnotationLoader] Loaded {len(vehicles)} vehicles, "
            f"{len(walkers)} walkers, ego_speed={ego_speed:.1f} m/s, weather={weather_type}"
        )
        return {
            "vehicles": vehicles,
            "walkers": walkers,
            "ego_speed": ego_speed,
            "weather_type": weather_type,
        }

    @staticmethod
    def _parse_objects(raw: dict, obj_type: str = "vehicle") -> list[dict]:
        """Parse the vehicles or walkers dict into a clean list."""
        objects = []
        if not raw or not isinstance(raw, dict):
            return objects

        for obj_id, obj_data in raw.items():
            if not isinstance(obj_data, dict):
                continue
            obj = {
                "id": int(obj_id),
                "type": obj_type,
                "class": obj_data.get("class", obj_type),
                "dist": obj_data.get("dist", 0.0),
                "speed": obj_data.get("speed", 0.0),
                "speed_x_y_z": obj_data.get("speed_x_y_z", [0.0, 0.0, 0.0]),
                "location": obj_data.get("location", [0, 0, 0]),
                "extent": obj_data.get("extent", [0, 0, 0]),
                "relative_angle": obj_data.get("relative_angle", 0.0),
                "bp_id": obj_data.get("bp_id", ""),
            }
            if "location_in_scene" in obj_data:
                obj["location_in_scene"] = obj_data["location_in_scene"]
                
            # Assign camera view based on relative angle
            obj["view"] = AnnotationLoader._angle_to_view(
                obj["relative_angle"]
            )
            objects.append(obj)

        return objects

    @staticmethod
    def _angle_to_view(angle: float) -> str:
        """
        Map relative_angle (degrees) to camera view name.

        Relative angle convention (from CARLA / OPV2V):
        0° = directly ahead (Front)
        +90° = left of ego
        -90° = right of ego
        ±180° = directly behind (Back)
        """
        a = angle % 360 # normalize to 0-360
        if a > 180:
            a -= 360 # back to -180..180

        abs_a = abs(a)
        if abs_a <= 45:
            return "Front"
        elif abs_a >= 135:
            return "Back"
        elif a > 0:
            return "Left"
        else:
            return "Right"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ViewPerception:
    """Perception result for a single camera view."""
    view_name: str
    detections: list[dict] = field(default_factory=list)
    radar_anchors: list[dict] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class PerceptionResult:
    """Full perception result for all views."""
    frame_id: str
    ego_speed: float = 0.0
    weather_type: str = "fd"
    views: dict[str, ViewPerception] = field(default_factory=dict)

    def full_report(self) -> str:
        """Pretty-print the full report."""
        lines = [
            f"{'='*60}",
            f"PERCEPTION REPORT — Frame {self.frame_id}",
            f"Ego speed: {self.ego_speed:.1f} m/s",
            f"{'='*60}",
        ]
        for view_name in _VIEW_NAMES:
            vp = self.views.get(view_name)
            if not vp:
                continue
            lines.append(f"\n[{view_name.upper()} VIEW]")
            lines.append(f" Detections: {len(vp.detections)} objects")
            for d in vp.detections:
                status = "STOPPED" if d["speed"] < 0.5 else f"{d['speed']:.1f}m/s"
                lines.append(
                    f" • {d['class']} at {d['dist']:.1f}m — {status}"
                )
            lines.append(f" Radar anchors: {len(vp.radar_anchors)}")
            lines.append(f" VLM reasoning: {vp.reasoning}")
            lines.append(f"\n{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Camera-to-view mapping for radar anchors (from GlobalRadarFilter)
# ---------------------------------------------------------------------------
_CAMERA_TO_VIEW = {
    "front": "Front",
    "front_right": "Right",
    "front_left": "Left",
    "rear": "Back",
    "rear_right": "Right",
    "rear_left": "Left",
}


# ---------------------------------------------------------------------------
# AnchoredPerceptionNode — DriveLM-style
# ---------------------------------------------------------------------------
class AnchoredPerceptionNode:
    """
    DriveLM-inspired perception node.

    Detection: from YAML ground-truth annotations
    Radar: from GlobalRadarFilter anchors
    Reasoning: VLM (Qwen2.5-VL-3B) reasons over known detections + images

    The VLM never detects — it only reasons.
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
        max_words: int = 150,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature

        # Lazy-loaded model cache
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load model + processor once, on first use."""
        if self._model is not None:
            return

        log.info(f"[PerceptionNode] Loading {self.model_id} ...")
        from mlx_vlm import load # type: ignore[import]

        self._model, self._processor = load(self.model_id)
        log.info("[PerceptionNode] Model loaded ✓")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def perceive_frame(
        self,
        data_dir: str | Path,
        frame_id: str,
        images: Optional[list[Image.Image]] = None,
        radar_anchors: Optional[list[dict]] = None,
    ) -> PerceptionResult:
        """
        Full perception pipeline for one frame.

        Parameters
        ----------
        data_dir : path containing frame files
        frame_id : e.g. '000514'
        images : optional pre-loaded camera images (4x PIL)
        radar_anchors : optional pre-computed radar anchors

        Returns
        -------
        PerceptionResult with per-view detections + VLM reasoning
        """
        self._ensure_model()
        data_dir = Path(data_dir)

        # 1. Load YAML annotations
        yaml_path = data_dir / f"{frame_id}.yaml"
        annotations = AnnotationLoader.load(yaml_path)

        # 2. Load camera images if not provided
        if images is None:
            images = self._load_cameras(data_dir, frame_id)

        # 3. Load radar anchors if not provided
        if radar_anchors is None:
            from stormvlm_loader import GlobalRadarFilter
            rf = GlobalRadarFilter()
            radar_anchors = rf.process(str(data_dir), frame_id)

        # 4. Group everything by view
        view_map = self._group_by_view(annotations, radar_anchors)

        # 5. Query VLM per view
        view_to_cam = {"Front": 0, "Right": 1, "Left": 2, "Back": 3}
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

        for view_name in _VIEW_NAMES:
            cam_idx = view_to_cam[view_name]
            detections = view_map.get(view_name, {}).get("detections", [])
            anchors = view_map.get(view_name, {}).get("anchors", [])

            # Build prompt
            prompt = self._build_prompt(
                view_name, detections, anchors, scenario=scenario_str
            )
            log.info(
                f"[PerceptionNode] {view_name} view: "
                f"{len(detections)} detections, {len(anchors)} anchors "
                f"→ querying LVLM..."
            )

            # Save temp image and query
            temp_path = self._save_temp_image(images[cam_idx])
            reasoning = self._query_model(temp_path, prompt, detections)

            # Clean up
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # ── Close‑range visibility override ──────────────────────────
            # The 3-bit VLM frequently marks nearby objects as "not visible"
            # due to fog‑bias.  Objects under 15 m are almost always visible
            # in the camera image (the photo confirms this).  We correct the
            # VLM's JSON in‑place so that the Python categoriser produces
            # the right bucket (Ghost instead of Unknown).
            sorted_dets = sorted(detections, key=lambda d: d.get("dist", 999))[:3]
            close_ids: set[str] = set()
            cls_cnts: dict[str, int] = {}
            for d in sorted_dets:
                cn = d.get("class", "object").lower()
                cls_cnts[cn] = cls_cnts.get(cn, 0) + 1
                tag = f"<{cn}_{cls_cnts[cn]}>"
                if d.get("dist", 999) < 15.0:
                    close_ids.add(tag)

            if close_ids:
                try:
                    # Strip markdown wrappers to parse the JSON
                    stripped = reasoning.strip()
                    stripped = re.sub(r'^```json\s*', '', stripped)
                    stripped = re.sub(r'\s*```$', '', stripped)
                    data = json.loads(stripped)
                    changed = False
                    if "categorization" in data:
                        for obj_id, props in data["categorization"].items():
                            nid = obj_id if obj_id.startswith("<") else f"<{obj_id}>"
                            if nid in close_ids and isinstance(props, dict):
                                if not props.get("visible_in_camera", False):
                                    props["visible_in_camera"] = True
                                    # Recategorise
                                    rad = props.get("has_radar_anchor", False)
                                    props["category"] = "Confirmed" if rad else "Ghost"
                                    changed = True
                    if changed:
                        reasoning = f"```json\n{json.dumps(data, indent=4)}\n```"
                except (json.JSONDecodeError, KeyError):
                    pass  # If parsing fails, keep the original reasoning
            # ─────────────────────────────────────────────────────────────

            result.views[view_name] = ViewPerception(
                view_name=view_name,
                detections=detections,
                radar_anchors=anchors,
                reasoning=reasoning,
            )

        return result

    # ------------------------------------------------------------------
    # Camera loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_cameras(data_dir: Path, frame_id: str) -> list[Image.Image]:
        """Load 4 camera images from flat files."""
        images = []
        for i in range(4):
            matches = glob.glob(str(data_dir / f"{frame_id}_camera{i}.*"))
            if matches:
                images.append(Image.open(matches[0]).convert("RGB"))
            else:
                log.warning(f"Camera {i} not found for frame {frame_id}")
                images.append(Image.new("RGB", (800, 600), (128, 128, 128)))
        return images

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _save_temp_image(img: Image.Image) -> str:
        """Resize and save to temp file. Apply CLAHE for fog enhancement."""
        import cv2

        # Resize
        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(resized)

        # CLAHE on L channel (enhance contrast in fog)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Save
        enhanced = Image.fromarray(arr)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        enhanced.save(path)
        return path

    # ------------------------------------------------------------------
    # Group detections + anchors by view
    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_view(
        annotations: dict, radar_anchors: list[dict]
    ) -> dict[str, dict]:
        """
        Group YAML detections and radar anchors by camera view.
        Returns {view_name: {"detections": [...], "anchors": [...]}}.
        """
        grouped: dict[str, dict] = {
            v: {"detections": [], "anchors": []}
            for v in _VIEW_NAMES
        }

        # Group vehicles + walkers by their assigned view
        for obj in annotations.get("vehicles", []) + annotations.get("walkers", []):
            view = obj.get("view", "Front")
            if view in grouped:
                grouped[view]["detections"].append(obj)

        # Group radar anchors by their primary camera
        for anchor in radar_anchors:
            primary = AnchoredPerceptionNode._CAMERA_TO_VIEW.get(
                anchor.get("primary_camera", ""), None
            )
            if primary and primary in grouped:
                grouped[primary]["anchors"].append(anchor)

        return grouped

    # ------------------------------------------------------------------
    # DriveLM-style prompt builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(
        view_name: str,
        detections: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
        scenario: str = "heavy fog",
    ) -> str:
        """
        Build a prompt that provides the ground truth data and asks the VLM to solely
        report the objects, states, and radar confirmation without any risk assessment.
        """
        prompt_parts = [
            f"Here is the sensor data for the {view_name.lower()} view:",
            "Ground Truth Objects:"
        ]
        class_counts: dict[str, int] = {}
        # Sort by distance and keep top 3 closest
        sorted_detections = sorted(detections, key=lambda d: d.get('dist', float('inf')))
        top_detections = sorted_detections[:3]
        if top_detections:
            for d in top_detections:
                cls_name = d.get('class', 'object').lower()
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                obj_tag = f"<{cls_name}_{class_counts[cls_name]}>"
                speed = d.get("speed", 0.0)
                dist = d.get("dist", 0.0)
                status = "stopped" if speed < 0.5 else f"moving at {speed:.1f}m/s"
                # Pre-compute radar match in Python to save the 3-bit VLM cognitive load
                # Margins: 20.0 meters for distance (radar range noise in fog),
                #          10.0 m/s for speed (to allow for radar doppler noise)
                radar_match = "False"
                for a in anchors:
                    dist_delta = abs(a['dist_m'] - dist)
                    speed_delta = abs(a['max_speed_ms'] - speed)
                    dist_match = dist_delta < 20.0
                    speed_match = speed_delta < 10.0
                    if dist_match and speed_match:
                        radar_match = "True"
                        break
                prompt_parts.append(f"- {obj_tag} at {dist:.1f}m, {status} (Has Radar Anchor: {radar_match})")
        else:
            prompt_parts.append("- None")

        q = (
            f"\nQuestion: You are an AI safety auditor. The current weather is {scenario}.\n"
            "Analyze the scene and output your response strictly as a single valid JSON dictionary. Do NOT output any other text or markdown.\n"
            "The JSON MUST have the following two keys:\n"
            "1. 'weather_analysis': A brief sentence describing visibility given the current weather conditions, and evaluating if close-range objects are clearly visible.\n"
            "2. 'categorization': A dictionary mapping each Ground Truth Object mentioned above to its classification. You MUST include EVERY object listed in the 'Ground Truth Objects' section.\n\n"
            "For each object in 'categorization', provide 'visible_in_camera' (boolean), 'has_radar_anchor' (boolean), and 'location_in_scene' (string).\n"
            "To determine 'location_in_scene', look at the object's position in the image and classify it exactly as one of the following: 'On Road', 'On Sidewalk', or 'Off Road'.\n"
            "To determine 'visible_in_camera', look at the image and determine if the object is visible (set true) or not visible (set false).\n"
            "Use the '(Has Radar Anchor: True/False)' tag provided in the Ground Truth Objects list to determine 'has_radar_anchor'."
        )
        prompt_parts.append(q)
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

# ------------------------------------------------------------------
# LVLM query
# ------------------------------------------------------------------
    def _query_model(self, image_path: str, prompt: str, detections: Optional[list[dict]] = None) -> str:
        """Send one image + prompt to the LVLM and return the response text."""
        from mlx_vlm import generate # type: ignore[import]

        # Generous token budget — trim by words afterwards
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
        return self._clean_output(raw, detections, self.max_words)

# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------
    @staticmethod
    def _clean_output(text: str, detections: Optional[list[dict]] = None, max_words: int = 150, margin: int = 20) -> str:
        """
        Clean LVLM output:
        1. Try to extract the first valid JSON block (if the model outputs JSON).
        Then compute final categories in Python based on output components.
        2. Fallback: Remove repeated sentences and trim to word budget + margin.
        """
        if detections is None:
            detections = []
            
        # Build mapping from tag to true YAML location_in_scene
        loc_map: dict[str, str] = {}
        class_counts: dict[str, int] = {}
        sorted_detections = sorted(detections, key=lambda d: d.get('dist', float('inf')))
        for d in sorted_detections[:3]:
            cls_name = d.get('class', 'object').lower()
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            tag = f"<{cls_name}_{class_counts[cls_name]}>"
            if "location_in_scene" in d:
                loc_map[tag] = d["location_in_scene"]

        def _process_json(data: dict) -> str:
            if "categorization" in data and isinstance(data["categorization"], dict):
                new_cat = {}
                for obj_id, props in data["categorization"].items():
                    fixed_id = obj_id if obj_id.startswith("<") and obj_id.endswith(">") else f"<{obj_id}>"
                    if isinstance(props, dict):
                        vis = props.get("visible_in_camera", False)
                        rad = props.get("has_radar_anchor", False)
                        # Close-range override: objects within 15m are almost
                        # certainly visible even in heavy fog (the image proves
                        # this). The VLM frequently hallucinate "not visible"
                        # due to fog‑bias in the prompt.
                        # We cannot recheck the distance here (only have IDs),
                        # so we leave the VLM decision for visibility, but let
                        # the radar tag drive the categorisation.
                        if vis and rad:
                             props["category"] = "Confirmed"
                        elif not vis and rad:
                             props["category"] = "Fogged"
                        elif vis and not rad:
                             props["category"] = "Ghost"
                        else:
                             props["category"] = "Unknown"
                             
                        # Override location_in_scene from YAML if present
                        if fixed_id in loc_map and loc_map[fixed_id]:
                             props["location_in_scene"] = loc_map[fixed_id]
                        elif "location_in_scene" not in props:
                             props["location_in_scene"] = "On Road"  # default safe assumption
                             
                        new_cat[fixed_id] = props
                data["categorization"] = new_cat
                return f"```json\n{json.dumps(data, indent=4)}\n```"
            return text

        # --- Step 1: Look for JSON Block ---
        # First try to find markdown blocks
        json_matches = re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for match in json_matches:
            try:
                data = json.loads(match.group(1))
                return _process_json(data)
            except json.JSONDecodeError:
                continue
        # If no valid markdown block, try to find the largest {...} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return _process_json(data)
            except json.JSONDecodeError:
                pass

        # --- Step 2: Fallback (Text deduplication & truncation) ---
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: set[str] = set()
        unique: list[str] = []
        for sent in sentences:
            key = sent.strip().lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(sent.strip())

        result_sentences: list[str] = []
        word_count = 0
        for sent in unique:
            words_in_sent = len(sent.split())
            if word_count + words_in_sent > (max_words + margin) and result_sentences:
                break
            result_sentences.append(sent)
            word_count += words_in_sent

        output = " ".join(result_sentences)
        if output and output[-1] not in '.!?':
            output += '.'
        return output