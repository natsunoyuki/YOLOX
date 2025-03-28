# Dog and cat detection dataset.
# https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "datasets/DogAndCat/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.class_names = ["dog", "cat"]
        self.num_classes = len(self.class_names)

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1
