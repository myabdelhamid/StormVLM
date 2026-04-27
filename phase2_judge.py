
"""
StormVLM — Phase 2: Mistral-7B Independent Audit
==================================================
Reads VLM outputs saved by phase1_generate.py and uses Mistral-7B-Instruct
(Mistral AI — different company) as an independent LLM-as-a-Judge to evaluate:

  1. Modality Evaluation:    Radar+Camera vs Camera-Only
  2. Representation Evaluation: JSON-Structured vs Text-Narrative

Outputs two clean CSV files with scores for every frame.
"""

import gc
import csv
import json
import re
from pathlib import Path

DATA_FILE = Path("evaluation_results/vlm_outputs.json")
MODALITY_CSV = Path("evaluation_results/modality_mistral_audit.csv")
REPRESENTATION_CSV = Path("evaluation_results/representation_mistral_audit.csv")
MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


def load_mistral():
    print(f"\n[Phase 2] Loading independent auditor: {MODEL_ID}...")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    print("[Phase 2] Mistral-7B loaded ✓")
    return model, tokenizer


def query_mistral(model, tokenizer, prompt: str, max_tokens: int = 400) -> str:
    from mlx_lm import generate
    chat = f"[INST] {prompt} [/INST]"
    result = generate(model, tokenizer, prompt=chat, max_tokens=max_tokens, temp=0.1, verbose=False)
    return result.strip()


def extract_json(text: str) -> dict:
    # Try markdown block
    for match in re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL):
        try:
            return json.loads(match.group(1))
        except Exception:
            continue
    # Try bare JSON object
    for match in re.finditer(r'\{.*?\}', text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except Exception:
            continue
    return {}


def build_modality_prompt(frame: dict) -> str:
    a = frame["pipeline_a_radar"]
    b = frame["pipeline_b_camera"]
    return (
        f"You are an expert autonomous driving safety auditor. "
        f"The ego vehicle is traveling at {frame['ego_speed']:.1f} m/s in {frame['weather']} conditions.\n\n"
        f"Two perception-planning pipelines produced different driving decisions for the same scene.\n\n"
        f"=== PIPELINE A (Radar + Camera Fusion) ===\n"
        f"Action: {a['action']}\n"
        f"Objects Tracked: {a['objects_tracked']}\n"
        f"Reasoning: {a['reasoning']}\n\n"
        f"=== PIPELINE B (Camera Only) ===\n"
        f"Action: {b['action']}\n"
        f"Objects Tracked: {b['objects_tracked']}\n"
        f"Reasoning: {b['reasoning']}\n\n"
        "Evaluate which pipeline made the safer and more logical decision. "
        "Radar fusion detects objects invisible to cameras in fog/rain.\n\n"
        "Respond ONLY with a JSON object containing exactly these keys:\n"
        '{"score_A": <int 1-10>, "score_B": <int 1-10>, "better_pipeline": <"A" or "B" or "Tie">, '
        '"justification": "<one sentence>"}\n\n'
        "JSON:"
    )


def build_representation_prompt(frame: dict) -> str:
    j = frame["pipeline_json_format"]
    t = frame["pipeline_text_format"]
    return (
        f"You are an expert evaluating autonomous driving AI systems. "
        f"Conditions: {frame['weather']}, ego speed {frame['ego_speed']:.1f} m/s.\n\n"
        f"Two pipelines processed the same driving scene using different output formats:\n\n"
        f"=== PIPELINE JSON (Structured JSON Output) ===\n"
        f"Action: {j['action']}\n"
        f"Reasoning: {j['reasoning']}\n"
        f"Raw Perception Sample: {j['raw_perception']}\n\n"
        f"=== PIPELINE TEXT (Natural Language Narrative) ===\n"
        f"Action: {t['action']}\n"
        f"Reasoning: {t['reasoning']}\n"
        f"Raw Perception Sample: {t['raw_perception']}\n\n"
        "Evaluate which output format produced safer, more coherent and more logically reasoned driving decisions. "
        "Consider: clarity of reasoning, safety of the action, and comprehensiveness.\n\n"
        "Respond ONLY with a JSON object containing exactly these keys:\n"
        '{"score_JSON": <int 1-10>, "score_TEXT": <int 1-10>, "better_format": <"JSON" or "TEXT" or "Tie">, '
        '"justification": "<one sentence>"}\n\n'
        "JSON:"
    )


def main():
    if not DATA_FILE.exists():
        print(f"[Phase 2] ERROR: {DATA_FILE} not found. Run phase1_generate.py first.")
        return

    with open(DATA_FILE) as f:
        data = json.load(f)
    frames = data.get("frames", [])
    print(f"\n[Phase 2] Loaded {len(frames)} frames from Phase 1.")

    # Load Mistral
    model, tokenizer = load_mistral()

    # Open CSV files
    with open(MODALITY_CSV, "w", newline="") as f1, open(REPRESENTATION_CSV, "w", newline="") as f2:
        writer_mod = csv.writer(f1)
        writer_mod.writerow(["Frame", "Scene", "Score_A (Radar)", "Score_B (Camera)", "Winner", "Justification"])

        writer_rep = csv.writer(f2)
        writer_rep.writerow(["Frame", "Scene", "Score_JSON_Format", "Score_Text_Format", "Winner", "Justification"])

        for i, frame in enumerate(frames):
            frame_id = frame["frame_id"]
            scene = frame["scene"]
            print(f"\n[{i+1}/{len(frames)}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"[{i+1}/{len(frames)}] JUDGING: Frame {frame_id} | Scene: {scene}")
            print(f"[{i+1}/{len(frames)}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            try:
                # 1. MODALITY EVALUATION
                mod_prompt = build_modality_prompt(frame)
                mod_raw = query_mistral(model, tokenizer, mod_prompt)
                mod_result = extract_json(mod_raw)
                writer_mod.writerow([
                    frame_id, scene,
                    mod_result.get("score_A"),
                    mod_result.get("score_B"),
                    mod_result.get("better_pipeline"),
                    mod_result.get("justification", "")
                ])
                f1.flush()

                # 2. REPRESENTATION EVALUATION
                rep_prompt = build_representation_prompt(frame)
                rep_raw = query_mistral(model, tokenizer, rep_prompt)
                rep_result = extract_json(rep_raw)
                writer_rep.writerow([
                    frame_id, scene,
                    rep_result.get("score_JSON"),
                    rep_result.get("score_TEXT"),
                    rep_result.get("better_format"),
                    rep_result.get("justification", "")
                ])
                f2.flush()

                print(f"")
                print(f"✅ [{i+1}/{len(frames)}] JUDGED — Frame {frame_id}")
                print(f"   Modality  : A={mod_result.get('score_A')} vs B={mod_result.get('score_B')} → Winner: {mod_result.get('better_pipeline')}")
                print(f"   Represent : JSON={rep_result.get('score_JSON')} vs TEXT={rep_result.get('score_TEXT')} → Winner: {rep_result.get('better_format')}")

            except Exception as e:
                print(f"    Error on {frame_id}: {e}")
                writer_mod.writerow([frame_id, scene, "ERROR", str(e)])
                writer_rep.writerow([frame_id, scene, "ERROR", str(e)])

    # Unload Mistral
    del model, tokenizer
    gc.collect()
    print(f"\n[Phase 2] Independent audit complete.")
    print(f"  → Modality results:        {MODALITY_CSV}")
    print(f"  → Representation results:  {REPRESENTATION_CSV}")

if __name__ == "__main__":
    main()
