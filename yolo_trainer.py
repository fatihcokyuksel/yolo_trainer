from ultralytics import YOLO
import time
from google.colab import files

def train(model_size="yolo11s.pt", epochs=50, image_size=640, batch_size=16, project_name="yolo_training"):
    model = YOLO(model_size)

    start_time = time.time()

    results = model.train(
        data="/content/data.yaml",
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project="/content/runs",
        name=project_name
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    best_mAP = results.results_dict.get("metrics/mAP50(B)", 0)

    print("\n" + "="*50)
    print(f"Eğitim Tamamlandı! \nEğitim süresi: {hours}sa {minutes}dk {seconds}sn")
    print(f"En iyi mAP@50: {best_mAP:.4f}")
    print(f"En iyi modelin konumu: /content/runs/{project_name}/weights/best.py")
    print("\n" + "="*50)

    print("\nModel indiriliyor...")
    files.download(f"/content/runs/{project_name}/weights/best.pt")


