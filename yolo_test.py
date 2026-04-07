from ultralytics import YOLO

model = YOLO("best.pt")

result = model.predict("test_img/resim4.png", save=True)

