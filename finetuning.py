import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import itertools
import numpy as np
import gc
from scipy import stats
import shutil 

if __name__ == "__main__":


    # --- 環境設定 --- 
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    ROOT_DIR = Path(__file__).parent
    DATA_YAML_PATH = ROOT_DIR / "datasets/data.yaml"
    PROJECT_BASE = ROOT_DIR / "finetuning_result"
    PROJECT_BASE.mkdir(exist_ok=True, parents=True)


    # --- モデルリスト --- 
    # MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    MODEL_NAMES = ["yolo11n"]


    # --- データオーグメンテーション条件リスト --- 
    mosaic_values = [0, 0.25,0.5]
    degrees_values = [0, 180.0]
    translate_values = [0, 0.25,0.5]
    perspective_values = [0, 0.0025,0.005]
    scale_values = [0, 0.25,0.5]
    
    

    augmentation_configs = [
        {"mosaic": m, "degrees": d, "translate": t, "perspective": p, "scale": s}
        for m, d, t, p, s in itertools.product(
            mosaic_values, degrees_values, translate_values, perspective_values, scale_values
        )
    ]

    print(f"Total augmentation combinations: {len(augmentation_configs)}")


    # --- プロジェクトフォルダ作成関数 --- 
    def get_project_path(config):
        folder_name = f"mosaic{config['mosaic']}_deg{config['degrees']}_trans{config['translate']}_pers{config['perspective']}_scale{config['scale']}"
        return Path(folder_name)


    # --- モデルごとの学習ループ --- 
    for model_name in MODEL_NAMES:
        print(f"\n========== Training model: {model_name} ==========")
        MODEL_PROJECT_BASE = PROJECT_BASE / model_name.split(".")[0]
        MODEL_PROJECT_BASE.mkdir(exist_ok=True, parents=True)

        results_summary = []

        for config in augmentation_configs:
            project_path = MODEL_PROJECT_BASE / get_project_path(config).name
            project_path.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Training with config: {config} ---")

            model = YOLO(model_name)

            # 学習
            model.train(
                data=str(DATA_YAML_PATH),
                lr0=0.001,
                epochs=50,
                batch=4,
                imgsz=640,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                mosaic=config["mosaic"],
                degrees=config["degrees"],
                translate=config["translate"],
                perspective=config["perspective"],
                scale=config["scale"],
                project=str(project_path),
                save=True,
                exist_ok=True,
                verbose=False,
                workers=0
            )

            metrics = model.val(data=str(DATA_YAML_PATH))

            if hasattr(metrics.box, 'ap'):
                ap = metrics.box.ap
                try:
                    mAP50 = float(ap[0]) if hasattr(ap, "__getitem__") else float(ap)
                except:
                    mAP50 = float(ap)

            if hasattr(metrics.box, 'f1'):
                try:
                    precision = float(getattr(metrics.box, "mp", 0))   
                    recall    = float(getattr(metrics.box, "mr", 0))   
                    f1_score  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                except:
                    f1_score = 0

            if hasattr(metrics, "confusion_matrix"):
                cm_raw  = metrics.confusion_matrix.matrix
                cm_normalized = cm_raw  / (cm_raw.sum(axis=0, keepdims=True) + 1e-9)
                confusion_matrix = cm_normalized
                cm = np.array(cm_normalized)
                TP=[]
                for i in range(cm.shape[0]-1):
                    TP.append(cm[i,i])
                tp_harmonic = stats.hmean(TP)

            results_summary.append({
                "config": config,
                "mAP50": mAP50,
                "F1": f1_score,
                "tp_harmonic": tp_harmonic,
                "confusion_matrix": confusion_matrix
            })

            del model
            gc.collect()

        # --- 結果のCSV保存 --- 
        df = pd.DataFrame([
            {
                "mosaic": r["config"]["mosaic"],
                "degrees": r["config"]["degrees"],
                "translate": r["config"]["translate"],
                "perspective":r["config"]["perspective"],
                "scale": r["config"]["scale"],
                "mAP50": r["mAP50"],
                "F1": r["F1"],
                "tp_harmonic": r["tp_harmonic"],
                
            }
            for r in results_summary
        ])
        df["avg_metric"] = df[["mAP50", "F1", "tp_harmonic"]].mean(axis=1)              
        df.to_csv(MODEL_PROJECT_BASE / "augmentation_summary.csv", index=False)
        
        best_idx = df["avg_metric"].idxmax()
        print(best_idx)
        best_row = df.loc[best_idx]
        best_row_dict = best_row.to_dict()
        best_row_dict = {k: int(v) if v == int(v) else v for k, v in best_row_dict.items()}
        print(best_row_dict)

        
        best_project_path = MODEL_PROJECT_BASE / get_project_path(best_row_dict)
        best_weights_path = best_project_path / "train"/"weights"/"best.pt"
        final_best_path = PROJECT_BASE / model_name.split(".")[0] / "best.pt"

        if best_weights_path.exists():
            shutil.copy2(best_weights_path, final_best_path)
            print(f"Best model copied to {final_best_path}")
            
            config_file_path = PROJECT_BASE /model_name.split(".")[0] / f"best_config_mosaic{best_row_dict['mosaic']}_deg{best_row_dict['degrees']}_trans{best_row_dict['translate']}_pers{best_row_dict['perspective']}_scale{best_row_dict['scale']}"
            config_file_path.touch()
            
        else:
            print("Error: best.pt does not exist! Copy aborted.")


