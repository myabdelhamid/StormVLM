"""
StormVLM — DriveLM-Style Prediction Node (Pk)
================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Architecture (inspired by DriveLM, ECCV 2024):
    Takes the filtered logical outputs of the Perception Node (PA) and 
    forecasts the future trajectory & risk level of validated objects.
"""

import json
import logging
import os
import re
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
log = logging.getLogger("stormvlm.prediction")


@dataclass
class PredictionResult:
    """Prediction result for the current frame."""
    frame_id: str
    prediction_json: str = ""

    def full_report(self) -> str:
        """Pretty-print the full report."""
        lines = [
            f"{'='*60}",
            f"PREDICTION REPORT — Frame {self.frame_id}",
            f"{'='*60}",
            self.prediction_json,
            f"\n{'='*60}"
        ]
        return "\n".join(lines)


class PredictionNode:
    """
    DriveLM-inspired prediction node.
    Takes categorized objects from PerceptionNode and assigns kinematic futures.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        temperature: float = 0.1,
        model=None,
        processor=None,
        max_words: int = 150,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self._model = model
        self._processor = processor
        self.max_words = max_words

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        log.info(f"[PredictionNode] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[PredictionNode] Model loaded ✓")

    def predict_frame(
        self,
        perception_result: PerceptionResult,
        data_dir: str | Path,
        images: Optional[list[Image.Image]] = None,
    ) -> PredictionResult:
        """Run the Phase 2 Prediction Node (Pk) using Perception outputs."""
        self._ensure_model()
        data_dir = Path(data_dir)

        # We need at least one camera image to anchor the Vision model.
        if images is None:
            import glob
            images = []
            matches = glob.glob(str(data_dir / f"{perception_result.frame_id}_camera0.*"))
            if matches:
                images.append(Image.open(matches[0]).convert("RGB"))
            else:
                images.append(Image.new("RGB", (800, 600), (128, 128, 128)))

        filtered_objects = []
        for view_name, vp in perception_result.views.items():
            json_match = re.search(r'```json\s*(.*?)\s*```', vp.reasoning, re.DOTALL)
            if not json_match:
                continue
            
            try:
                data = json.loads(json_match.group(1))
                cat_dict = data.get("categorization", {})
            except json.JSONDecodeError:
                continue
                
            class_counts: dict[str, int] = {}
            top_detections = sorted(vp.detections, key=lambda d: d.get('dist', float('inf')))[:3]
            det_map = {}
            for d in top_detections:
                cls_name = d.get('class', 'object').lower()
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                obj_tag = f"<{cls_name}_{class_counts[cls_name]}>"
                det_map[obj_tag] = d

            for obj_id, props in cat_dict.items():
                category = props.get("category", "Unknown")
                if category == "Unknown":
                    continue
                
                if obj_id in det_map:
                    d = det_map[obj_id]
                    speed = d.get("speed", 0.0)
                    dist = d.get("dist", 0.0)
                    speed_xyz = d.get("speed_x_y_z", [0.0, 0.0, 0.0])
                    vx = speed_xyz[0]
                    ego_speed = perception_result.ego_speed
                    v_rel_x = vx - ego_speed
                    
                    if abs(vx) < 0.5 and abs(speed_xyz[1]) < 0.5:
                        motion = "Stationary"
                    elif view_name == "Front":
                        motion = "Approaching (Closing in)" if v_rel_x < 0 else "Moving away (Diverging)"
                    elif view_name == "Back":
                        motion = "Approaching from behind (Closing in)" if v_rel_x > 0 else "Moving away (Falling behind)"
                    else:
                        motion = "Moving forwards relative to ego" if v_rel_x > 0 else "Moving backwards relative to ego"

                    motion_hint = motion
                    
                    clean_id = obj_id.strip("<>")
                    
                    filtered_objects.append({
                        "id": f"{view_name}_{clean_id}",
                        "category": category,
                        "location_in_scene": props.get("location_in_scene", "On Road"),
                        "view": view_name,
                        "dist_m": round(dist, 1),
                        "speed_ms": round(d.get("speed", 0.0), 1),
                        "v_rel_x": round(v_rel_x, 1),
                        "motion_hint": motion_hint
                    })

        prompt = self._build_prediction_prompt(filtered_objects)
        
        # Save placeholder right image to pass mlx generation restrictions
        temp_path = self._save_temp_image(images[0])
        log.info(f"[PredictionNode] Querying LVLM with {len(filtered_objects)} valid objects...")
        reasoning = self._query_model(temp_path, prompt)
        
        try:
            os.unlink(temp_path)
        except OSError:
            pass

        # Post-process: fix contradictory VLM risk/motion outputs
        reasoning = self._postprocess_predictions(reasoning, filtered_objects)
            
        return PredictionResult(
            frame_id=perception_result.frame_id,
            prediction_json=reasoning
        )

    def _save_temp_image(self, img: Image.Image) -> str:
        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        resized.save(path)
        return path

    @staticmethod
    def _postprocess_predictions(
        reasoning: str, filtered_objects: list[dict]
    ) -> str:
        """
        Fix contradictory VLM outputs using physics-based rules:
        1. Objects moving away → always NO RISK.
        2. Off-road objects → always NO RISK.
        3. Speed > 0.5 m/s but marked "Stationary" → fix motion description.
        4. Close-range objects ahead → bump risk if too low.
        5. Enrich generic reasoning with actual kinematic data.
        """
        # Build a lookup from filtered_objects for distance/speed data
        obj_lookup: dict[str, dict] = {}
        for o in filtered_objects:
            obj_lookup[o["id"]] = o

        # Parse the VLM JSON
        clean = reasoning.strip()
        clean = re.sub(r'^```json\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return reasoning  # Can't fix what we can't parse

        predictions = []
        if isinstance(data, list):
            predictions = data
        elif isinstance(data, dict) and "scene_prediction" in data:
            predictions = data["scene_prediction"]
        else:
            return reasoning

        for pred in predictions:
            obj_id = pred.get("id", "")
            motion = pred.get("future_motion", "").lower()
            risk = pred.get("risk_level", "LOW").upper()
            src = obj_lookup.get(obj_id, {})
            speed = src.get("speed_ms", 0.0)
            dist = src.get("dist_m", 999.0)
            view = src.get("view", "")
            
            # Enforce the true category from the perception input (don't trust VLM)
            true_category = src.get("category", pred.get("category", "Unknown"))
            pred["category"] = true_category

            # ── Rule 1: Moving away = always NO RISK ──────────────────────
            away_keywords = ["diverging", "falling behind", "moving away"]
            if any(kw in motion for kw in away_keywords):
                pred["risk_level"] = "NO RISK"
                pred["kinematic_reasoning"] = (
                    f"Object at {dist}m is moving away at {speed}m/s — "
                    f"no collision risk."
                )
                continue

            # ── Rule 2: Off-road objects are NO RISK ─────
            location = src.get("location_in_scene", "On Road")
            pred["location_in_scene"] = location
            if location in ["On Sidewalk", "Off Road"]:
                pred["risk_level"] = "NO RISK"
                pred["kinematic_reasoning"] = (
                    f"Object is {location.lower()} — "
                    f"no collision risk."
                )
                continue

            # ── Rule 3: Fix "Stationary" when speed > 0.5 ────────────
            if "stationary" in motion and speed > 0.5:
                if view == "Front":
                    pred["future_motion"] = f"Slow-moving at {speed}m/s ahead"
                else:
                    pred["future_motion"] = f"Slow-moving at {speed}m/s"

            # ── Rule 3: Close-range objects ahead need higher risk ─────
            if view == "Front" and dist < 15.0:
                if true_category in ("Confirmed", "Fogged"):
                    if "stationary" in pred.get("future_motion", "").lower() or speed < 0.5:
                        pred["risk_level"] = "HIGH"
                        pred["kinematic_reasoning"] = (
                            f"Radar-confirmed object stopped at {dist}m directly "
                            f"ahead — imminent collision risk."
                        )
                    elif risk == "LOW":
                        pred["risk_level"] = "MODERATE"
                        pred["kinematic_reasoning"] = (
                            f"Radar-confirmed object at {dist}m ahead moving at "
                            f"{speed}m/s — potential hazard."
                        )
                elif true_category == "Ghost":
                    # Visible but no radar — still close, so at least MODERATE
                    if speed < 0.5:
                        pred["risk_level"] = "MODERATE"
                        pred["kinematic_reasoning"] = (
                            f"Visible stationary object at {dist}m ahead without "
                            f"radar — possible real obstacle."
                        )
                    elif risk == "LOW":
                        pred["risk_level"] = "MODERATE"
                        pred["kinematic_reasoning"] = (
                            f"Visible object at {dist}m moving at {speed}m/s — "
                            f"needs monitoring."
                        )

            # ── Rule 4: Enrich generic reasoning ──────────────────────
            reasoning_text = pred.get("kinematic_reasoning", "")
            if reasoning_text and "no risk" in reasoning_text.lower() and dist < 20.0:
                pred["kinematic_reasoning"] = (
                    f"Object at {dist}m with speed {speed}m/s — "
                    f"close proximity requires attention."
                )

        # Re-wrap
        if isinstance(data, list):
            return f"```json\n{json.dumps(data, indent=4)}\n```"
        else:
            data["scene_prediction"] = predictions
            return f"```json\n{json.dumps(data, indent=4)}\n```"


    @staticmethod
    def _build_prediction_prompt(filtered_objects: list[dict]) -> str:
        prompt_parts = [
            "System Task: You are the StormVLM Prediction Engine. Predict the future motion and risk level of each object.",
            "Rules:",
            "- Use the provided 'motion_hint' (approaching/diverging) to understand the object's trajectory relative to the ego vehicle.",
            "- Fogged objects are confirmed physical threats hidden by weather.",
            "- Ghost objects lack radar data; assume they are stationary obstacles.",
            "",
            "Input Data:"
        ]
        
        if not filtered_objects:
            prompt_parts.append("[]")
        else:
            prompt_parts.append(json.dumps(filtered_objects, indent=4))
            
        prompt_parts.extend([
            "",
            "Output format: Return ONLY a valid JSON block matching the scene_prediction schema, WRAPPED IN ```json ... ``` markdown.",
            "Do NOT output any metadata, system tags, or explanations. Stop immediately after the closing ```.",
            "The JSON MUST represent the 'Input Data' provided. Return a flat list of objects under 'scene_prediction'.",
            "The JSON structure MUST be:",
            "```json",
            "{",
            '    "scene_prediction": [',
            '        {',
            '            "id": "<id>",',
            '            "category": "Fogged/Ghost/Confirmed",',
            '            "speed": "e.g., 7.9m/s",',
            '            "future_motion": "Describe future trajectory based on motion_hint.",',
            '            "kinematic_reasoning": "Reason about risk using dist and v_rel_x.",',
            '            "risk_level": "LOW/MODERATE/HIGH"',
            '        }',
            '    ]',
            "}",
            "```"
        ])
        
        return "\n".join(prompt_parts)

    def _query_model(self, image_path: str, prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]
        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=1000,
            temperature=self.temperature,
            verbose=False,
            resize_shape=_MAX_IMAGE_SIZE,
        )
        raw = result.text.strip() if hasattr(result, "text") else str(result).strip()
        return self._clean_prediction_output(raw, self.max_words)

    @staticmethod
    def _clean_prediction_output(text: str, max_words: int = 150, margin: int = 20) -> str:
        """
        Clean Prediction output:
          1. Extract ONLY the first valid JSON block.
          2. Fallback: Remove repeated sentences and trim to word budget + margin.
        """
        # 1. Try to find markdown blocks
        json_matches = re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for match in json_matches:
            content = match.group(1).strip()
            # If there's nested markdown, stay safe
            if "```json" in content:
                content = content.split("```json")[-1].strip()
            
            try:
                data = json.loads(content)
                return f"```json\n{json.dumps(data, indent=4)}\n```"
            except json.JSONDecodeError:
                continue
                
        # 2. Non-greedy fallback for raw JSON blocks
        # We find all { ... } blocks and take the FIRST one that is a valid scene_prediction
        raw_blocks = re.finditer(r'(\{.*?\})', text, re.DOTALL)
        for block_match in raw_blocks:
            try:
                block = block_match.group(1)
                data = json.loads(block)
                if "scene_prediction" in data:
                    return f"```json\n{json.dumps(data, indent=4)}\n```"
            except json.JSONDecodeError:
                continue

        # 3. Final attempt: Brute force finding the largest valid JSON sub-block
        best_candidate: Optional[str] = None
        start_idx = text.find('{')
        while start_idx != -1:
            end_idx = text.rfind('}', start_idx)
            while end_idx != -1 and end_idx > start_idx:
                try:
                    candidate = text[start_idx:end_idx + 1]
                    data = json.loads(candidate)
                    if "scene_prediction" in data:
                        # Heuristic: Check if the JSON actually contains objects (list not empty)
                        pred = data["scene_prediction"]
                        if isinstance(pred, list) and len(pred) > 0:
                            return f"```json\n{json.dumps(data, indent=4)}\n```"
                        # If no content yet, save it as a candidate but keep looking
                        best_candidate = candidate
                except json.JSONDecodeError:
                    pass
                # Move end_idx backwards to find the next possible closing brace
                end_idx = text.rfind('}', start_idx, end_idx)
            # Move start_idx forwards to find the next possible opening brace
            start_idx = text.find('{', start_idx + 1)

        if best_candidate:
             data = json.loads(best_candidate)
             return f"```json\n{json.dumps(data, indent=4)}\n```"

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
            if key in ["```json", "```"]:
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
        
        # Final deduplication of the entire string if it contains the same block twice
        if len(output) > 20:
            mid = len(output) // 2
            first_half = output[:mid].strip()
            second_half = output[mid:].strip()
            if first_half in second_half or second_half in first_half:
                return first_half if len(first_half) > len(second_half) else second_half

        return output
