import random
import os
import sys
import shutil
import argparse
from pathlib import Path


DATA_PATH = "/content/dataset"
TRAIN_PCT = 0.9
SEED = 42


if not os.path.isdir(DATA_PATH):
    print("DATA_PATH Bulunamadı! Bir önceki hücrede data.zip dosyasının doğru açıldığından emin olun.")
    sys.exit(0)

if not (0.01 <= TRAIN_PCT <= 0.99):
    print("TRAIN_PCT 0.01 ile 0.99 arasında olmalı!")
    sys.exit(0)


output_base = "/content/data"
if os.path.exists(output_base):
    shutil.rmtree(output_base)
    print("Eski train/val klasörleri temizlendi!")


train_img_path = os.path.join(output_base, "train/images")
train_lbl_path = os.path.join(output_base, "train/labels")
val_img_path = os.path.join(output_base, "val/images")
val_lbl_path = os.path.join(output_base, "val/labels")

for path in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
    os.makedirs(path)



input_img_path = os.path.join(DATA_PATH, "images")
input_lbl_path = os.path.join(DATA_PATH, "labels")

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

img_files = [
    p for p in Path(input_img_path).rglob("*")
    if p.suffix.lower() in image_exts
]

lbl_files = {p.stem for p in Path(input_lbl_path).rglob("*.txt")}

matched = [p for p in img_files if p.stem in lbl_files]
no_label = [p for p in img_files if p.stem not in lbl_files]
no_image = lbl_files - {p.stem for p in img_files}


if no_label:
    print(f"Etiketsiz görüntü: {len(no_label)} --- background image olarak kopyalanacak")

if no_image:
    print(f"Görüntüsüz etiket: {len(no_image)} --- atlanacak!")

random.seed(SEED)
random.shuffle(img_files)

train_num = int(len(img_files) * TRAIN_PCT)
train_files = img_files[:train_num]
val_files = img_files[train_num:]

print(f"Train: {len(train_files)} görüntü")
print(f"Validation: {len(val_files)} görüntü")



def copy_files(file_list, img_dest, lbl_dest, input_lbl_path):
    for img_path in file_list:
        lbl_path = Path(input_lbl_path) / (img_path.stem + ".txt")
        shutil.copy(img_path, os.path.join(img_dest, img_path.name))
        if lbl_path.exists():
            shutil.copy(lbl_path, os.path.join(lbl_dest, lbl_path.name))




copy_files(train_files, train_img_path, train_lbl_path, input_lbl_path)
copy_files(val_files, val_img_path, val_lbl_path, input_lbl_path)

shutil.copy(
    os.path.join(DATA_PATH, "classes"),
    os.path.join(output_base, "classes")
)

print("Train/Validation bölümü tamamlandı! Bir sonraki hücreye geçebilirsin!")



