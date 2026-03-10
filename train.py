# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import warnings
warnings.filterwarnings('ignore')
import os.path as osp
import random
import numpy as np
import sys
import os
# sys.path.append("/")
# os.chdir("..")
import collections
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from clip_cc import datasets
from clip_cc.models.cm import ClusterMemory
from clip_cc.trainers import VITFP16
from clip_cc.evaluators import Evaluator, extract_features
from clip_cc.utils.data import IterLoader
from clip_cc.utils.data import transforms as T
from clip_cc.utils.data.preprocessor import Preprocessor
from clip_cc.utils.logging import Logger
from clip_cc.utils.serialization import load_checkpoint, save_checkpoint
from clip_cc.utils.faiss_rerank import compute_jaccard_distance
from clip_cc.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clip_cc.models.model_clip import make_model
from clip_cc.utils.prepare_optimizer import make_vit_optimizer
from clip_cc.utils.prepare_scheduler import create_scheduler
from datetime import datetime

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    sampler = RandomMultipleGallerySampler(train_set, num_instances)

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, mutual=False),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=False, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    # model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
    #                       num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model = make_model()

    # use CUDA
    model.cuda()

    model = nn.DataParallel(model)

    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def adaptive_eps(dist_matrix, rho=0.005):
    tri_mat = dist_matrix[np.triu_indices(dist_matrix.shape[0], 1)]
    tri_mat_sorted = np.sort(tri_mat)
    top_num = int(np.round(rho * len(tri_mat_sorted)))
    eps = tri_mat_sorted[:top_num].mean()
    return eps

def multi_scale_dbscan(dist_matrix, eps_values, min_samples=4):
    all_labels = []
    for eps in eps_values:
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(dist_matrix)
        all_labels.append(labels)

    all_labels = np.array(all_labels)
    final_labels = -1 * np.ones(all_labels.shape[1], dtype=int)

    for i in range(all_labels.shape[1]):
        labels_i = all_labels[:, i]
        valid_labels = labels_i[labels_i != -1]

        if len(valid_labels) == 0:
            final_labels[i] = -1
        elif len(set(valid_labels)) == 1:
            final_labels[i] = valid_labels[0]
        else:
            # 邻居约束融合：查找最近的邻居并决定类别
            nearest_idxs = np.argsort(dist_matrix[i])[:min_samples]
            neighbor_labels = all_labels[:, nearest_idxs].flatten()
            neighbor_labels = neighbor_labels[neighbor_labels != -1]

            if len(neighbor_labels) > 0:
                final_labels[i] = collections.Counter(neighbor_labels).most_common(1)[0][0]
            else:
                final_labels[i] = -1

    return final_labels

def relabel_pseudo_labels(labels):
    valid_labels = sorted(set(labels) - {-1})
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    new_labels = np.array([label_mapping[label] if label != -1 else -1 for label in labels])
    num_clusters = len(label_mapping)
    return new_labels, num_clusters

def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = False
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters > 0) else None
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    model = create_model(args)
    evaluator = Evaluator(model)
    optimizer = make_vit_optimizer(model)
    lr_scheduler = create_scheduler(optimizer)
    trainer = VITFP16(model)

    best_mAP = 0.0

    for epoch in range(args.epochs):
        print('=> EPOCH num={}'.format(epoch + 1))
        with torch.no_grad():
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print('==> Extract features for clustering...')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))
            features, _ = extract_features(model, cluster_loader)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            # 根据数据集动态选择聚类策略
            if args.dataset.lower() == 'prcc':
                eps_values = [
                    adaptive_eps(rerank_dist, rho=0.002),
                    adaptive_eps(rerank_dist, rho=0.004),
                    adaptive_eps(rerank_dist, rho=0.006),
                ]
                print(f'Multi-scale eps for prcc: {eps_values}')
                pseudo_labels = multi_scale_dbscan(rerank_dist, eps_values, min_samples=5)
            else:
                eps_small = adaptive_eps(rerank_dist, rho=0.003)
                eps_large = adaptive_eps(rerank_dist, rho=0.006)
                print(f'Multi-scale eps: eps_small={eps_small:.3f}, eps_large={eps_large:.3f}')
                pseudo_labels = multi_scale_dbscan(rerank_dist, [eps_small, eps_large], min_samples=4)



            pseudo_labels, num_cluster = relabel_pseudo_labels(pseudo_labels)
            num_outliers = (pseudo_labels == -1).sum()

            print(f'Clusters after relabeling: {num_cluster}, Outliers: {num_outliers}, Outlier ratio: {num_outliers / len(pseudo_labels):.2%}')

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[label].append(features[i])
            centers = [torch.stack(centers[idx]).mean(0) for idx in sorted(centers.keys())]
            return torch.stack(centers, dim=0)

        cluster_features = generate_cluster_features(pseudo_labels, features)
        del cluster_loader, features

        memory = ClusterMemory(1280, num_cluster, temp=args.temp, momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        trainer.memory = memory

        pseudo_labeled_dataset = [(fname, label.item(), cid, clothid)
                                  for (fname, _, cid, clothid), label in zip(sorted(dataset.train), pseudo_labels) if label != -1]

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, num_instances=16, iters=iters,
                                        trainset=pseudo_labeled_dataset)

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'=> Current Lr: {curr_lr:.2e}')

        time.sleep(0.5)
        train_loader.new_epoch()
        time.sleep(0.5)

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}   best: {:5.1%}{}\n'.
                  format(epoch+1, mAP,  best_mAP, ' *' if is_best else ''))
        lr_scheduler.step()

        torch.cuda.empty_cache()
        print('=> CUDA cache is released.')

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='msmt17',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.7,  # market 0.6 msmt 0.7
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="/home/ykding/dataset/general-reid")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
