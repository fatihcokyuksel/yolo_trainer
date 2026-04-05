import os
import zipfile

ZIP_NAME = "data.zip"

UPLOAD_PATH = f"/content/{ZIP_NAME}"
EXTRACT_PATH = "/content/dataset"

if not os.path.exists(UPLOAD_PATH):
    raise FileNotFoundError(f"{ZIP_NAME} bulunamadı. Dosyayı yüklediğinizden emin olunuz.")



with zipfile.ZipFile(UPLOAD_PATH, 'r',) as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)



required = {
    "images/": os.path.isdir(os.path.join(EXTRACT_PATH, "images")),
    "labels/": os.path.isdir(os.path.join(EXTRACT_PATH, "labels")),
    "classes": os.path.isfile(os.path.join(EXTRACT_PATH, "classes.txt")),
    "notes.json": os.path.isfile(os.path.join(EXTRACT_PATH, "notes.json"))
}


errors = []
warnings = []

for name, exists in required.items():
    if not exists:
        errors.append(name)



images_dir = os.path.join(EXTRACT_PATH, "images")
labels_dir = os.path.join(EXTRACT_PATH, "labels")

if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    }

    label_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.endswith(".txt")
    }

    image_count = len(image_stems)
    label_count = len(label_stems)



    missing_labels = image_stems - label_stems
    if missing_labels:
        warnings.append(
            f"{len(missing_labels)} görüntünün etiketi yok: "
            + ", ".join(sorted(missing_labels)[:5])
            + (" ..." if len(missing_labels) > 5 else "")
        )

    
    missing_images = label_stems - image_stems
    if missing_images:
        warnings.append(
            f"{len(missing_images)} etiketin görüntüsü yok: "
            + ", ".join(sorted(missing_images)[:5])
            + (" ..." if len(missing_images) > 5 else "")
        )



    
    

if errors:
    print("DOĞRULAMA BAŞARISIZ!")
    for e in errors:
        print(f"-- {e}")
    print("\n Zip dosyalarının yapısını kontrol edip tekrar yükle!")
    
elif warnings:
    print("DOĞRULAMA BAŞARISIZ!")
    for w in warnings:
        print(f"-- {w}")
    print("Her verinin etiketli olduğunu veya her etiketin verisi olduğunu kontrol et!")

else:
    classes_path = os.path.join(EXTRACT_PATH, "classes.txt")
    if os.path.isfile(classes_path):
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"Sınıflar: {', '.join(classes)}")
    print("\n" + "="*50)
    print("Tüm kontroller başarılı! Br sonraki hücreye geçebilirsin!")
    
    







