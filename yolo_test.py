from ultralytics import YOLO

model = YOLO("best.pt")

result = model.predict("resim3.png", save=True)

