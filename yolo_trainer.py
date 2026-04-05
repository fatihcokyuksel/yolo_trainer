from ultralytics import YOLO

def train(model_size="yolov11s.pt", epochs=50, image_size=640, batch_size=16, project_name="yolo_training"):
    model = YOLO(model_size)
    model.train(
        data="/content/data.yaml",
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project="/content/runs",
        name=project_name
    )

    print("\nEğitim Tamamlandı!")
    print(f"En iyi model: /content/runs/{project_name}/weights/best.pt")




