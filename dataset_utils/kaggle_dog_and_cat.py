# Download the data as a zipfile from
# https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection
# unzip it, and place the unzipped folder  `archive` under the directory 
# `datasets/` and run this script from the terminal.

from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import json
import random

random.seed(0)


DATA_DIR = "datasets/"
ORIG_SUB_DIR = "archive/"
SUB_DIR = "DogAndCat/"
ANNOTATIONS_DIR = "annotations/"
ORIG_IMG_DIR = "images/"
ORIG_TRAIN_DIR = "train"
ORIG_VAL_DIR = "valid"
ORIG_ANN_FILE = "_annotations.coco.json"
TRAIN_DIR = "train2017"
VAL_DIR = "val2017"
ANN_FILE = "instances_"


def xml_to_dict(xml_path):
    # Decode the .xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Return the image size, object label and bounding box 
    # coordinates together with the filename as a dict.
    return {"filename": xml_path,
            "image_width": int(root.find("./size/width").text),
            "image_height": int(root.find("./size/height").text),
            "image_channels": int(root.find("./size/depth").text),
            "label": root.find("./object/name").text,
            "x1": int(root.find("./object/bndbox/xmin").text),
            "y1": int(root.find("./object/bndbox/ymin").text),
            "x2": int(root.find("./object/bndbox/xmax").text),
            "y2": int(root.find("./object/bndbox/ymax").text)}


# TODO improve the script. This was made on the fly to test things out...
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()


    # Rename the unzipped folder `archive/` to `DogAndCat/`.
    if (script_dir.parent.resolve() / DATA_DIR / ORIG_SUB_DIR).exists():
        (script_dir.parent.resolve() / DATA_DIR / ORIG_SUB_DIR).rename(
            script_dir.parent.resolve() / DATA_DIR / SUB_DIR
        )

    data_dir = script_dir.parent.resolve() / DATA_DIR / SUB_DIR


    # List all individual files in the annotations directory.
    ann_files = sorted(list((data_dir / ANNOTATIONS_DIR).iterdir()))
    
    train_files = random.sample(ann_files, int(len(ann_files)*0.8))
    val_files = list(set(ann_files).difference(set(train_files)))
    #train_files = ann_files[:int(len(ann_files)*0.8)]
    #val_files = ann_files[int(len(ann_files)*0.8):]

    train_images = []
    for f in train_files:
        if f.name == ".DS_Store":
            train_files.remove(f)

    for f in train_files:
        if (data_dir / ORIG_IMG_DIR / (f.stem + ".png")).exists():
            train_images.append((data_dir / ORIG_IMG_DIR / (f.stem + ".png")))
        elif (data_dir / ORIG_IMG_DIR / (f.stem + ".jpg")).exists():
            train_images.append((data_dir / ORIG_IMG_DIR / (f.stem + ".jpg")))

    assert len(train_files) == len(train_images)
    #print(set([f.stem for f in train_files]).difference(set([f.stem for f in train_images])))

    val_images = []
    for f in val_files:
        if f.name == ".DS_Store":
            val_files.remove(f)
    
    for f in val_files:
            if (data_dir / ORIG_IMG_DIR / (f.stem + ".png")).exists():
                val_images.append((data_dir / ORIG_IMG_DIR / (f.stem + ".png")))
            elif (data_dir / ORIG_IMG_DIR / (f.stem + ".jpg")).exists():
                val_images.append((data_dir / ORIG_IMG_DIR / (f.stem + ".jpg")))

    assert len(val_files) == len(val_images)        


    # Make train annotations.
    train_ann_json = {
        'info': {
            'year': '2024',
            'version': '1',
            'description': 'Kaggle dog and cat detection dataset.',
            'contributor': '',
            'url': '',
            'date_created': '2024-01-01'},
        'licenses': [{'id': 1, 'url': 'https://creativecommons.org/licenses/by/4.0/', 'name': 'CC BY 4.0'}], 
        'categories': [{'id': 1, 'name': 'dog'}, {'id': 2, 'name': 'cat'}], 
        'images': [], 
        'annotations': [],
    }
    
    rev_img_id_ann = {}
    for i, f in enumerate(train_images):
        img = cv2.imread(str(f))
        train_ann_json["images"].append(
            {
                "id": i,
                "license": 1,
                "file_name": f.name,
                "height": img.shape[0],
                "width": img.shape[1],
                "date_captured": "2024-01-01"
            }
        )
        rev_img_id_ann[f.stem] = i

    train_anns = []
    for f in train_files:
        train_anns.append(xml_to_dict(f))

    for i, f in enumerate(train_anns):
        train_ann_json["annotations"].append(
            {
                'id': i,
                'image_id': rev_img_id_ann[f["filename"].stem],
                'category_id': 1 if f["label"] == "dog" else 2,
                'bbox': [f["x1"], f["y1"], f["x2"] - f["x1"], f["y2"] - f["y1"]],
                'area': (f["x2"] - f["x1"]) * (f["y2"] - f["y1"]),
                'segmentation': [[]],
                'iscrowd': 0,
            }
        )


    # Make val annotations.
    val_ann_json = {
        'info': {
            'year': '2024',
            'version': '1',
            'description': 'Kaggle dog and cat detection dataset.',
            'contributor': '',
            'url': '',
            'date_created': '2024-01-01'},
        'licenses': [{'id': 1, 'url': 'https://creativecommons.org/licenses/by/4.0/', 'name': 'CC BY 4.0'}], 
        'categories': [{'id': 1, 'name': 'dog'}, {'id': 2, 'name': 'cat'}], 
        'images': [], 
        'annotations': [],
    }
    
    rev_img_id_ann = {}
    for i, f in enumerate(val_images):
        img = cv2.imread(str(f))
        val_ann_json["images"].append(
            {
                "id": i,
                "license": 1,
                "file_name": f.name,
                "height": img.shape[0],
                "width": img.shape[1],
                "date_captured": "2024-01-01"
            }
        )
        rev_img_id_ann[f.stem] = i

    val_anns = []
    for f in val_files:
        val_anns.append(xml_to_dict(f))

    for i, f in enumerate(val_anns):
        val_ann_json["annotations"].append(
            {
                'id': i,
                'image_id': rev_img_id_ann[f["filename"].stem],
                'category_id': 1 if f["label"]=="dog" else 2,
                'bbox': [f["x1"], f["y1"], f["x2"] - f["x1"], f["y2"] - f["y1"]],
                'area': (f["x2"] - f["x1"]) * (f["y2"] - f["y1"]),
                'segmentation': [[]],
                'iscrowd': 0,
            }
        )


    # Remove all .xml files
    for f in (data_dir / ANNOTATIONS_DIR).iterdir():
        if f.suffix == ".xml":
            (data_dir / ANNOTATIONS_DIR / f).unlink(missing_ok=True)


    # Make JSON files.
    with open(data_dir / ANNOTATIONS_DIR / (ANN_FILE + VAL_DIR + ".json"), "w") as f:
        json.dump(val_ann_json, f, indent=4)

    with open(data_dir / ANNOTATIONS_DIR / (ANN_FILE + TRAIN_DIR + ".json"), "w") as f:
        json.dump(train_ann_json, f, indent=4)


    # Copy images to correct sub directories.
    (data_dir / TRAIN_DIR).mkdir(exist_ok=True)
    for f in (data_dir / ORIG_IMG_DIR).iterdir():
        if f in train_images:
            f.rename(data_dir / TRAIN_DIR / f.name)

    (data_dir / ORIG_IMG_DIR).rename(data_dir / VAL_DIR)
