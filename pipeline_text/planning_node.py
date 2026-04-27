"""
StormVLM — Natural Language Planning Node (Pl-Text)
=====================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Ablation Study: Structured JSON vs. Natural Language Text

This node is the TEXT COUNTERPART of PlanningNode. It receives the
prediction output and ego speed, then selects a driving action and
provides reasoning — entirely through VLM natural language generation.

KEY DESIGN DIFFERENCE from the JSON version:
  The JSON PlanningNode uses DETERMINISTIC Python rules to select the action,
  then asks the VLM only for a justification sentence.
  This text node gives the VLM the SAME physics rules as prompt instructions
  and lets the VLM perform both action selection AND reasoning in one step.

  This tests whether a VLM can reliably follow safety-critical decision rules
  when they are given as natural language instructions vs. hard-coded in Python.

Differences from planning_node.py (JSON version):
  • Action selection: VLM-driven (with strict rules in prompt) vs. Python code
  • Output format: structured paragraph ending with "FINAL ACTION: X"
  • No json.loads(), no JSON parsing
  • No _select_action() Python method — rules are in the prompt

Similarities (controlled variables):
  ✓ Same action space (KEEP_SPEED, DECELERATE, STOP, EMERGENCY_BRAKE, etc.)
  ✓ Same physics rules (just expressed as prompt instructions)
  ✓ Same model, temperature
  ✓ Same risk priority logic
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Optional

from PIL import Image  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.planning_t")

# Same action space as JSON version
ACTIONS = [
    "KEEP_SPEED",
    "ACCELERATE",
    "DECELERATE",
    "STOP",
    "EMERGENCY_BRAKE",
    "STEER_TO_AVOID",
]


class PlanningNodeText:
    """
    Natural-language planning node (ablation counterpart of PlanningNode).

    Same physics rules, same action space — but the VLM makes the decision
    using natural language Chain-of-Thought reasoning instead of Python code.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        max_words: int = 200,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature
        self._model = model
        self._processor = processor

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[PlanningNodeText] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[PlanningNodeText] Model loaded ✓")

    # ── Public entry point ────────────────────────────────────────────────
    def plan_action(
        self, prediction_text: str, ego_speed: float
    ) -> dict:
        """
        Generate a driving plan from prediction text.

        Returns dict with:
          - selected_action: extracted from "FINAL ACTION: X"
          - planning_reasoning: full VLM reasoning text
        """
        self._ensure_model()

        prompt = self._build_planning_prompt(prediction_text, ego_speed)

        try:
            from mlx_vlm import generate  # type: ignore[import]

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
                max_tokens=300,
                temperature=self.temperature,
                verbose=False,
            )
            raw = (
                result.text.strip()
                if hasattr(result, "text")
                else str(result).strip()
            )

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Clean text output
            reasoning = self._clean_text_output(raw, self.max_words)

            # Extract the FINAL ACTION from the text
            action = self._extract_action(reasoning)

            return {
                "selected_action": action,
                "planning_reasoning": reasoning,
            }

        except Exception as e:
            log.warning(f"[PlanningNodeText] VLM failed ({e}); using fallback.")
            return {
                "selected_action": "DECELERATE",
                "planning_reasoning": f"VLM inference failed: {e}. "
                f"Defaulting to DECELERATE for safety. "
                f"FINAL ACTION: DECELERATE",
            }

    # ══════════════════════════════════════════════════════════════════════
    # Planning Prompt — TEXT VERSION
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _build_planning_prompt(prediction_text: str, ego_speed: float) -> str:
        """
        Build a highly optimized NATURAL LANGUAGE planning prompt.

        Same physics rules as the Python _select_action() method in
        planning_node.py — but expressed as Chain-of-Thought instructions.
        """
        # Truncate prediction text if too long
        pred_words = prediction_text.split()
        if len(pred_words) > 300:
            prediction_text = ' '.join(pred_words[:300]) + '...'

        prompt = (
            "You are the StormVLM Planning Engine — the final decision-maker "
            "in an autonomous driving safety pipeline. Your job is to select "
            "the single safest driving action based on the prediction analysis below.\n"
            "\n"
            f"## Current State\n"
            f"- Ego vehicle speed: {ego_speed:.1f} m/s\n"
            f"- {'The vehicle is currently STOPPED.' if ego_speed < 0.01 else f'The vehicle is moving at {ego_speed:.1f} m/s.'}\n"
            "\n"
            "## Prediction Analysis\n"
            f"{prediction_text}\n"
            "\n"
            "## Available Actions\n"
            "You MUST select exactly one of these actions:\n"
            "- **KEEP_SPEED**: Maintain current speed. Use when the path is clear.\n"
            "- **ACCELERATE**: Increase speed. Use only when stopped and path is clear.\n"
            "- **DECELERATE**: Reduce speed gradually. Use for MODERATE risk threats ahead.\n"
            "- **STOP**: Come to a controlled stop. Use for confirmed obstacles close ahead.\n"
            "- **EMERGENCY_BRAKE**: Maximum braking force. Use ONLY for imminent collision "
            "(HIGH risk CONFIRMED or FOGGED object within 15m directly ahead).\n"
            "- **STEER_TO_AVOID**: Lateral evasion. Use when braking alone is insufficient.\n"
            "\n"
            "## Mandatory Decision Rules (YOU MUST FOLLOW THESE)\n"
            "Apply these rules in order:\n"
            "\n"
            "1. **Only FRONT objects affect braking decisions.** Objects behind (Back_) "
            "or to the sides (Left_, Right_) NEVER trigger braking or stopping — that "
            "would cause a rear-end collision.\n"
            "\n"
            "2. **If no FRONT objects have risk above NO RISK:**\n"
            "   - If stopped → ACCELERATE\n"
            "   - If moving → KEEP_SPEED\n"
            "\n"
            "3. **If a FRONT object is DIVERGING (moving away) → ignore it.** "
            "It cannot collide with the ego vehicle.\n"
            "\n"
            "4. **If the ego vehicle is already STOPPED (speed ≈ 0):**\n"
            "   - HIGH/MODERATE risk ahead → KEEP_SPEED (stay stopped, wait)\n"
            "   - LOW or no risk → ACCELERATE\n"
            "   - NEVER decelerate or brake when already stopped.\n"
            "\n"
            "5. **For moving ego vehicle with FRONT threats:**\n"
            "   - HIGH risk + FOGGED/CONFIRMED ahead → EMERGENCY_BRAKE\n"
            "   - HIGH risk + GHOST ahead → EMERGENCY_BRAKE\n"
            "   - MODERATE risk ahead → DECELERATE\n"
            "   - LOW risk + high ego speed (>5 m/s) → DECELERATE\n"
            "   - LOW risk + low ego speed → KEEP_SPEED\n"
            "\n"
            "## Your Response Format\n"
            "Write a concise safety analysis in plain English:\n"
            "1. First, identify the most critical object (highest risk, closest, ahead).\n"
            "2. Explain your reasoning using the rules above.\n"
            "3. End your response with the following line on its own:\n"
            "\n"
            "FINAL ACTION: [YOUR_CHOSEN_ACTION]\n"
            "\n"
            "CRITICAL: Do NOT use JSON, code blocks, or dictionaries. "
            "Write in plain English only. The last line MUST be "
            "\"FINAL ACTION:\" followed by one of the six available actions in ALL CAPS.\n"
        )
        prompt += "\nAnswer:"

        return prompt

    # ── Extract action from text ─────────────────────────────────────────
    @staticmethod
    def _extract_action(text: str) -> str:
        """
        Extract the FINAL ACTION from the VLM's text output.
        Looks for 'FINAL ACTION: <ACTION>' pattern.
        """
        # Primary: look for "FINAL ACTION: X"
        match = re.search(
            r'FINAL\s+ACTION\s*:\s*(\w[\w_\s]*)',
            text, re.IGNORECASE
        )
        if match:
            action = match.group(1).strip().upper()
            # Normalize common variants
            action = action.replace(" ", "_")
            if action in (
                "KEEP_SPEED", "ACCELERATE", "DECELERATE",
                "STOP", "EMERGENCY_BRAKE", "STEER_TO_AVOID",
                "MAINTAIN_SPEED",
            ):
                if action == "MAINTAIN_SPEED":
                    return "KEEP_SPEED"
                return action

        # Fallback: look for action keywords in the last 50 words
        last_words = ' '.join(text.split()[-50:]).lower()

        if "emergency brak" in last_words or "emergency_brake" in last_words:
            return "EMERGENCY_BRAKE"
        if "steer to avoid" in last_words or "steer_to_avoid" in last_words:
            return "STEER_TO_AVOID"
        if "come to a stop" in last_words or "full stop" in last_words:
            return "STOP"
        if "decelerate" in last_words or "slow down" in last_words or "reduce speed" in last_words:
            return "DECELERATE"
        if "accelerate" in last_words or "speed up" in last_words:
            return "ACCELERATE"
        if "maintain" in last_words or "keep speed" in last_words or "keep_speed" in last_words:
            return "KEEP_SPEED"

        log.warning(
            "[PlanningNodeText] Could not extract action; defaulting to DECELERATE."
        )
        return "DECELERATE"

    # ── Text cleanup ─────────────────────────────────────────────────────
    @staticmethod
    def _clean_text_output(text: str, max_words: int = 200) -> str:
        """Clean VLM output: strip artifacts, deduplicate, enforce budget."""
        # Strip code fences
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

        # Enforce word budget (but protect the FINAL ACTION line)
        words = text.split()
        if len(words) > max_words:
            # Check if FINAL ACTION is near the end
            full_text = ' '.join(words)
            action_match = re.search(r'FINAL\s+ACTION\s*:.*', full_text, re.IGNORECASE)
            if action_match:
                # Keep the action line even if over budget
                before_action = full_text[:action_match.start()]
                action_line = action_match.group(0)
                before_words = before_action.split()
                if len(before_words) > max_words - 10:
                    before_action = ' '.join(before_words[:max_words - 10])
                    if before_action and before_action[-1] not in '.!?\n':
                        before_action += '.'
                text = before_action + '\n\n' + action_line
            else:
                text = ' '.join(words[:max_words])
                if text and text[-1] not in '.!?':
                    text += '.'

        return text if text else "No planning analysis available."
