#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

python inference_clip_cc.py \
    -d prcc \
    --data-dir /home/ykding/dataset/general-reid \
    --checkpoint /path/to/model_best.pth.tar
"""
import argparse
import os
import os.path as osp
import sys
import time
import random
import numpy as np
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from clip_cc import datasets
from clip_cc.evaluators import Evaluator
from clip_cc.models.model_clip import make_model
from clip_cc.utils.data import transforms as T
from clip_cc.utils.data.preprocessor import Preprocessor
from clip_cc.utils.logging import Logger
from clip_cc.utils.serialization import load_checkpoint


def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer,
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transform),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True
    )
    return loader


def create_model():
    model = make_model()
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Clip-CC Re-ID inference")
    parser.add_argument('-d', '--dataset', type=str, default='ltcc', choices=datasets.names())
    parser.add_argument('--data-dir', type=str, default='/home/ykding/dataset')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint', type=str,
                        default='/home/ykding/code4/code-2025-2/3-unsuper/PCL-CLIP-USL-main-c/log/ltcc/model_best-293.pth.tar')
    parser.add_argument('--logs-dir', type=str, default='./logs_infer')

    args = parser.parse_args()

    #
    # seed = random.randint(0, 99999)
    seed = 87477
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print(f"Using random seed: {seed}")

    if not osp.exists(args.checkpoint):
        sys.exit(f'✗ Checkpoint not found: {args.checkpoint}')

    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.logs_dir, f'{args.dataset}_infer.log'))

    print('Inference configuration:')
    for k, v in vars(args).items():
        print(f'{k:>12}: {v}')

    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    model = create_model()

    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'=> Loaded checkpoint {args.checkpoint}')

    evaluator = Evaluator(model)

    print('Starting evaluation...')
    start_time = time.monotonic()

    mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

    elapsed_time = timedelta(seconds=time.monotonic() - start_time)

    print('\nFinal Results:')
    print(f'Mean AP: {mAP:.1%}')
    print(f'Time cost: {elapsed_time}')


if __name__ == '__main__':
    main()
