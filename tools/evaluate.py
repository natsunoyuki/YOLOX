from pathlib import Path
import yaml
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)


class EvaluateConfigs:
    def __init__(self, yaml_args):
        self.experiment_name = yaml_args.get("experiment_name", None)
        self.name = yaml_args.get("name", self.experiment_name)

        self.dist_backend = yaml_args.get("dist_backend", "nccl")
        self.dist_url = yaml_args.get("dist_url", None)

        self.batch_size = yaml_args.get("batch_size", 16)
        self.devices = yaml_args.get("devices", None)
        self.num_machines = yaml_args.get("num_machines", 1)
        self.machine_rank = yaml_args.get("machine_rank", 0)
        
        self.exp_file = yaml_args.get("exp_file", None)

        self.ckpt = yaml_args.get("ckpt", None)

        self.conf = yaml_args.get("conf", 0.25)
        self.nms = yaml_args.get("nms", 0.3)
        self.tsize = yaml_args.get("tsize", None)
        self.seed = yaml_args.get("seed", None)

        self.fp16 = yaml_args.get("fp16", False)
        self.legacy = yaml_args.get("legacy", False)
        self.fuse = yaml_args.get("fuse", False)
        self.trt = yaml_args.get("trt", False)

        self.test = yaml_args.get("test", False)
        self.speed = yaml_args.get("speed", False)
        self.opts = yaml_args.get("opts", [])


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    configure_module()

    with open(script_dir / "train.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = EvaluateConfigs(args)

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
