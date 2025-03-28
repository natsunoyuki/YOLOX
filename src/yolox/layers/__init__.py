#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# import torch first to make jit op work without `ImportError of libc10.so`
import torch  # noqa

from yolox.layers.jit_ops import FastCOCOEvalOp, JitOp

try:
    from yolox.layers.fast_coco_eval_api import COCOeval_opt
except ImportError:  #  exception will be raised when users build yolox from source
    pass
