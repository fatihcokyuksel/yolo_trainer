import yaml
import os


def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    if not os.path.exists(path_to_classes_txt):
        print(f"classes.txt bulunamadı, şu konumda olmalı: {path_to_classes_txt}")
        return
    
    with open(path_to_classes_txt, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    number_of_classes = len(classes)

    data = {
        "path": "/content/data",
        "train": "train/images",
        "val": "val/images",
        "nc": number_of_classes,
        "names": classes
    }

    with open(path_to_data_yaml, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"data.yaml oluşturuldu: {path_to_data_yaml}")
    print(f"sınıf sayısı: {number_of_classes}")
    print(f"Sınıflar: {', '.join(classes)}")





    

