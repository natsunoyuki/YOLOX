# SkyFusion: Aerial Object Detection
# A synthetic dataset generated fromAITODv2 and Airbus Aircraft Detection
# https://www.kaggle.com/datasets/kailaspsudheer/tiny-object-detection
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/SkyFusion_aircraft/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.class_names = ["aircraft"]
        self.num_classes = len(self.class_names)

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1
