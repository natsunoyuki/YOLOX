<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >


## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities. For more details, please refer to the [report on Arxiv](https://arxiv.org/abs/2107.08430).

This repository is an implementation of PyTorch version YOLOX, there is also a [MegEngine implementation](https://github.com/MegEngine/YOLOX).

This repository is a fork of the original by [Megvii] (https://github.com/Megvii-BaseDetection/YOLOX), and contains new improvements and modifications.

<img src="assets/git_fig.png" width="1000" >


## Benchmarks
#### Standard Models.
|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

#### Light Models.
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |


## Installation
Clone this repository and install locally with a virtual environment.
```bash
git clone https://github.com/natsunoyuki/YOLOX
cd YOLOX

python3 -m venv venv
source venv/bin/activate
pip3 install -e .
```

## YOLOX Train, Evaluate and Predict Tools
Various scripts for training, evaluating and predicting are available under `tools/`, while the corresponding configuration YAML files should be placed under `tools_configs/`. A default set of configuration files are available under `tools_configs/`, and additional files may be created and passed to the script as an argument.

### Training Data Preparation
By convention, all training data should be in the <a href = "https://cocodataset.org/#home">MS-COCO</a> format. The images for the train, validation and test splits should be placed under individual folders `train2017/`, `val2017/` and `test2017/`, and all annotation JSON files should be placed under `annotations/` as in the tree below. The object detection bounding box annotations should be in the <a href = "https://cocodataset.org/#home">MS-COCO</a> format: `[x0, y0, w, h]`.

```bash
datasets/
├── data_dir_1
├── ...
└── data_dir_N
    ├── train2017/
    │   └── *.jpg
    ├── val2017/
    │   └── *.jpg
    ├── test2017/
    │   └── *.jpg
    └── annotations
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── instances_test2017.json
```

### Model Training
After placing the training data and "exp" file (see section below) in the appropriate locations, set up the training configurations in `tools_configs/<data-name>/train.yaml`, and run the training script:
```bash
python3 tools/train.py --config_dir tools_configs/<data-name>/ --config train.yaml
```

### Model Evaluation
After placing the evaluation data and "exp" file (see section below) in the appropriate locations, set up the evaluation configurations in `tools_config/<data-name>/evaluate.yaml`, and run the evaluation script:
```bash
python3 tools/evaluate.py --config_dir tools_configs/<data-name>/ --config evaluate.yaml
```

### Predict
After placing the prediction data and "exp" file (see section below) in the appropriate locations, set up the prediction configurations in `tools_config/<data-name>/predict.yaml`, and run the prediction script:
```bash
python3 tools/predict.py --config_dir tools_configs/<data-name>/ --config predict.yaml
```


## Exp Files
Settings for YOLOX models are defined in "exp" `.py` files under `exps/`. These files contain model settings such as the model depth and width, the number of classes in the data, the location of the data, the number of training epochs etc. A new "exp" file should be created for each new experiment, which will be used for training, evaluating and predicting with the model. Some example "exp" files are provided under `exps/default/` and `exps/example/custom/`. Additionally, some other "exp" files for datasets found on Kaggle such as `exps/example/kaggle_dog_and_cat/` are also available as examples on how to craft "exp" files for custom data.


## Deployment
1. [MegEngine in C++ and Python](./demo/MegEngine)
2. [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
3. [TensorRT in C++ and Python](./demo/TensorRT)
4. [ncnn in C++ and Java](./demo/ncnn)
5. [OpenVINO in C++ and Python](./demo/OpenVINO)
6. [Accelerate YOLOX inference with nebullvm in Python](./demo/nebullvm)


## Third-party resources
* YOLOX for streaming perception: [StreamYOLO (CVPR 2022 Oral)](https://github.com/yancie-yjr/StreamYOLO)
* The YOLOX-s and YOLOX-nano are Integrated into [ModelScope](https://www.modelscope.cn/home). Try out the Online Demo at [YOLOX-s](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox/summary) and [YOLOX-Nano](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox_nano_coco/summary) respectively 🚀.
* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sultannn/YOLOX-Demo)
* The ncnn android app with video support: [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* YOLOX with Tengine support: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX Deploy DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX MNN/TNN/ONNXRuntime: [YOLOX-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolox.cpp)、[YOLOX-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolox.cpp) and [YOLOX-ONNXRuntime C++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)


## Cite YOLOX
If you use YOLOX in your research, please cite the original authors' work by using the following BibTeX entry:
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```


## In memory of Dr. Jian Sun
Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our remembrance and condolences to our captain Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center"><img src="assets/sunjian.png" width="200"></div>
