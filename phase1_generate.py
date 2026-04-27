
"""
StormVLM — Phase 1: VLM Generation
====================================
Runs all Qwen2.5-VL inference across the 50 approved frames and saves
every output to a JSON file. After this script finishes, the VLM is
fully unloaded from memory so Mistral-7B can be loaded in Phase 2.
"""

import os
import gc
import json
from pathlib import Path
from pipeline_json.perception_node import AnchoredPerceptionNode, AnnotationLoader
from pipeline_json.prediction_node import PredictionNode
from pipeline_json.planning_node import PlanningNode
from pipeline_text.perception_node import PerceptionNodeText
from pipeline_text.prediction_node import PredictionNodeText
from pipeline_text.planning_node import PlanningNodeText
from stormvlm_loader import GlobalRadarFilter

DATA_ROOT = Path("/Users/marwannelsayed/Desktop/M/Master's Shahzad/Datasets_Master")
OUTPUT_FILE = Path("evaluation_results/vlm_outputs.json")

SAMPLED_FRAMES = [
    ("train_three/unj_hrd_d/389", "000100"), ("train_four/rsnj_fd_s/137", "000532"),
    ("train_three/unj_srd_d/431", "000292"), ("train_four/rsnj_fd_s/179", "000536"),
    ("train_four/rsnj_fhrd_d/179", "000804"), ("train_one/ri_fhrd_d/193", "000066"),
    ("train_four/rsnj_fd_d/179", "000822"), ("train_four/rsnj_fd_s/179", "000162"),
    ("train_three/unj_hrd_d/389", "000110"), ("train_four/rsnj_fhrd_d/137", "000282"),
    ("train_four/rsnj_fhrd_s/179", "000490"), ("train_one/ri_fhrd_s/161", "000256"),
    ("train_four/rsnj_fhrd_d/137", "000170"), ("train_four/rsnj_fd_s/179", "000068"),
    ("train_four/rsnj_fd_d/158", "000596"), ("train_three/unj_srd_d/431", "000120"),
    ("train_four/rsnj_fd_s/158", "000376"), ("train_one/ri_fhrd_d/193", "000264"),
    ("train_four/rsnj_fhrd_d/214", "000500"), ("train_three/unj_cd_s/271", "000272"),
    ("train_four/rsnj_fhrd_s/137", "000452"), ("train_four/rsnj_fd_d/137", "000532"),
    ("train_four/unj_fhrd_s/271", "000150"), ("train_one/ri_fhrd_s/137", "000060"),
    ("train_three/unj_fd_d/410", "000250"), ("train_four/rsnj_fd_d/179", "000144"),
    ("train_one/ri_fd_d/193", "000518"), ("train_four/rsnj_fd_s/179", "000162"),
    ("train_three/unj_hrd_d/431", "000304"), ("train_four/unj_fd_d/431", "000510"),
    ("train_one/ri_fd_d/235", "000548"), ("train_three/unj_cd_s/271", "000272"),
    ("train_four/rsnj_fhrd_s/137", "000452"), ("train_four/rsnj_fd_d/137", "000532"),
    ("train_four/unj_fhrd_s/271", "000150"), ("train_one/ri_fhrd_d/214", "000144"),
    ("train_four/rsnj_fhrd_d/179", "000392"), ("train_one/ri_fhrd_s/137", "000060"),
    ("train_three/unj_fd_d/410", "000250"), ("train_one/ri_fd_d/214", "000086"),
    ("train_four/rsnj_fhrd_d/137", "000200"), ("train_three/unj_hrd_d/193", "000100"),
    ("train_one/ri_fhrd_d/193", "000442"), ("train_four/rsnj_fd_d/179", "000822"),
    ("train_one/ri_fd_d/193", "000518"), ("train_three/unj_hrd_d/431", "000304"),
    ("train_four/unj_fd_d/431", "000510"), ("train_one/ri_fd_d/235", "000548"),
    ("train_three/unj_cd_s/271", "000272"), ("train_four/rsnj_fhrd_s/137", "000452")
]

def main():
    Path("evaluation_results").mkdir(exist_ok=True)

    # Load existing results for resume support
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = {r["key"]: r for r in json.load(f).get("frames", [])}
    
    print(f"\n[Phase 1] VLM Generation — {len(SAMPLED_FRAMES)} frames")
    print(f"[Phase 1] Already completed: {len(existing)} frames. Resuming...")

    # Initialize VLM
    perc_json = AnchoredPerceptionNode()
    perc_json._ensure_model()
    model, processor = perc_json._model, perc_json._processor
    
    nodes = {
        "json_pred": PredictionNode(model=model, processor=processor),
        "json_plan": PlanningNode(model=model, processor=processor),
        "text_perc": PerceptionNodeText(model_id=perc_json.model_id),
        "text_pred": PredictionNodeText(model_id=perc_json.model_id),
        "text_plan": PlanningNodeText(model=model, processor=processor),
    }
    nodes["text_perc"]._model, nodes["text_perc"]._processor = model, processor
    nodes["text_pred"]._model, nodes["text_pred"]._processor = model, processor
    rf = GlobalRadarFilter()

    results = list(existing.values())

    for i, (rel_scene, frame_id) in enumerate(SAMPLED_FRAMES):
        key = f"{rel_scene}::{frame_id}"
        if key in existing:
            print(f"[{i+1}/50] Skipping (already done): {frame_id} in {rel_scene}")
            continue

        scene_path = DATA_ROOT / rel_scene
        print(f"\n[{i+1}/50] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{i+1}/50] GENERATING: Frame {frame_id} | Scene: {rel_scene}")
        print(f"[{i+1}/50] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            yaml_path = scene_path / f"{frame_id}.yaml"
            anns = AnnotationLoader.load(yaml_path)
            weather = anns.get("weather_type", "fog")

            # Pipeline A: Radar + Camera
            res_a = perc_json.perceive_frame(data_dir=scene_path, frame_id=frame_id, radar_anchors=None)
            p_a = nodes["json_pred"].predict_frame(res_a, scene_path)
            pl_a = nodes["json_plan"].plan_action(p_a.prediction_json, res_a.ego_speed)

            # Pipeline B: Camera Only
            res_b = perc_json.perceive_frame(data_dir=scene_path, frame_id=frame_id, radar_anchors=[])
            p_b = nodes["json_pred"].predict_frame(res_b, scene_path)
            pl_b = nodes["json_plan"].plan_action(p_b.prediction_json, res_b.ego_speed)

            # Pipeline C: Text Narrative
            res_t = nodes["text_perc"].perceive_frame(data_dir=scene_path, frame_id=frame_id)
            p_t = nodes["text_pred"].predict_frame(res_t, scene_path)
            pl_t = nodes["text_plan"].plan_action(p_t.prediction_text, res_t.ego_speed)

            record = {
                "key": key,
                "frame_id": frame_id,
                "scene": rel_scene,
                "weather": weather,
                "ego_speed": res_a.ego_speed,
                "pipeline_a_radar": {
                    "action": pl_a.get("selected_action", ""),
                    "reasoning": pl_a.get("planning_reasoning", "")[:500],
                    "objects_tracked": len(nodes["json_plan"]._parse_prediction(p_a.prediction_json)),
                },
                "pipeline_b_camera": {
                    "action": pl_b.get("selected_action", ""),
                    "reasoning": pl_b.get("planning_reasoning", "")[:500],
                    "objects_tracked": len(nodes["json_plan"]._parse_prediction(p_b.prediction_json)),
                },
                "pipeline_json_format": {
                    "action": pl_a.get("selected_action", ""),
                    "reasoning": pl_a.get("planning_reasoning", "")[:500],
                    "raw_perception": str(list(res_a.views.values())[0].reasoning)[:400] if res_a.views else "",
                },
                "pipeline_text_format": {
                    "action": pl_t.get("selected_action", ""),
                    "reasoning": pl_t.get("planning_reasoning", "")[:500],
                    "raw_perception": str(list(res_t.views.values())[0].reasoning)[:400] if res_t.views else "",
                },
            }
            results.append(record)

            # Save after every frame
            with open(OUTPUT_FILE, "w") as f:
                json.dump({"frames": results}, f, indent=2)

            print(f"")
            print(f"✅ [{i+1}/50] COMPLETED — Frame {frame_id}")
            print(f"   Radar Action : {record['pipeline_a_radar']['action']}")
            print(f"   Camera Action: {record['pipeline_b_camera']['action']}")
            print(f"   Text Action  : {record['pipeline_text_format']['action']}")
            print(f"   Saved to: {OUTPUT_FILE}")

        except Exception as e:
            print(f"    Error on {frame_id}: {e}")

    # Unload VLM from memory
    print("\n[Phase 1] All VLM inference complete.")
    print("[Phase 1] Unloading VLM from memory...")
    del perc_json, model, processor, nodes, rf
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    print("[Phase 1] VLM fully unloaded. Memory freed.")
    print(f"[Phase 1] Results saved to: {OUTPUT_FILE}")
    print("[Phase 1] → Now run: python phase2_judge.py")

if __name__ == "__main__":
    main()
