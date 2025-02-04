# Download the data as a zipfile, unzip it, and place the unzipped folder 
# `SkyFusion` under the directory `datasets/` and run this script from
# the terminal.

from pathlib import Path
import shutil


DATA_DIR = "datasets/"
SUB_DIR = "SkyFusion/"
ANNOTATIONS_DIR = "annotations/"
ORIG_TEST_DIR = "test"
ORIG_TRAIN_DIR = "train"
ORIG_VAL_DIR = "valid"
ORIG_ANN_FILE = "_annotations.coco.json"
TEST_DIR = "test2017"
TRAIN_DIR = "train2017"
VAL_DIR = "val2017"
ANN_FILE = "instances_"


# TODO improve the script. This was made on the fly to test things out...
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    data_dir = script_dir.parent.resolve() / DATA_DIR / SUB_DIR


    # Make `annotations/` sub-directory.
    (data_dir / ANNOTATIONS_DIR).mkdir(exist_ok=True)


    # Rename the annotations JSON files.
    if (data_dir / ORIG_TEST_DIR / ORIG_ANN_FILE).exists():
        (data_dir / ORIG_TEST_DIR / ORIG_ANN_FILE).rename(
            data_dir / ORIG_TEST_DIR / (ANN_FILE + TEST_DIR + ".json")
        )

    if (data_dir / ORIG_TRAIN_DIR / ORIG_ANN_FILE).exists():
        (data_dir / ORIG_TRAIN_DIR / ORIG_ANN_FILE).rename(
            data_dir / ORIG_TRAIN_DIR / (ANN_FILE + TRAIN_DIR + ".json")
        )

    if (data_dir / ORIG_VAL_DIR / ORIG_ANN_FILE).exists():
        (data_dir / ORIG_VAL_DIR / ORIG_ANN_FILE).rename(
            data_dir / ORIG_VAL_DIR / (ANN_FILE + VAL_DIR + ".json")
        )


    # Then move renamed annotation files to `annotations/`
    if (data_dir / ORIG_TEST_DIR / (ANN_FILE + TEST_DIR + ".json")).exists():
        shutil.move(
            (data_dir / ORIG_TEST_DIR / (ANN_FILE + TEST_DIR + ".json")),
            (data_dir / ANNOTATIONS_DIR / (ANN_FILE + TEST_DIR + ".json")),
        )

    if (data_dir / ORIG_TRAIN_DIR / (ANN_FILE + TRAIN_DIR + ".json")).exists():
        shutil.move(
            (data_dir / ORIG_TRAIN_DIR / (ANN_FILE + TRAIN_DIR + ".json")),
            (data_dir / ANNOTATIONS_DIR / (ANN_FILE + TRAIN_DIR + ".json")),
        )

    if (data_dir / ORIG_VAL_DIR / (ANN_FILE + VAL_DIR + ".json")).exists():
        shutil.move(
            (data_dir / ORIG_VAL_DIR / (ANN_FILE + VAL_DIR + ".json")),
            (data_dir / ANNOTATIONS_DIR / (ANN_FILE + VAL_DIR + ".json")),
        )

    # Then rename the image sub dirs.
    if (data_dir / ORIG_TEST_DIR).exists():
        (data_dir / ORIG_TEST_DIR).rename(data_dir / TEST_DIR)

    if (data_dir / ORIG_TRAIN_DIR).exists():
        (data_dir / ORIG_TRAIN_DIR).rename(data_dir / TRAIN_DIR)

    if (data_dir / ORIG_VAL_DIR).exists():
        (data_dir / ORIG_VAL_DIR).rename(data_dir / VAL_DIR)
