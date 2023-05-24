# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import imp
import os
import warnings
from numbers import Number
from collections import defaultdict
import os.path as osp
import platform
import multiprocessing as mp

import cv2
import mmcv
import mmengine
import numpy as np
import torch

from torch.nn import DataParallel

from mmengine.config import DictAction
from mmengine.runner import load_checkpoint
from mmengine.registry import DefaultScope
from mmengine.runner import Runner
from mmpretrain.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('out', help='output result file')
    parser.add_argument('--threshold', default=None, type=float, help='open-set threshold')
    parser.add_argument('--no-scores', action='store_true', help='don\'t write score .csv file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--device', default=None, help='device used for testing. (Deprecated)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()

    #assert args.metrics or args.out, \
    #    'Please specify at least one of output path and evaluation metrics.'

    return args


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmengine.ProgressBar(len(dataset))
    observation_ids = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            #data = model.module.data_preprocessor(data, training=False)
            imgs = data['inputs'].cuda()
            result = model.module.extract_feat(imgs)

        filenames = [x.img_path for x in data['data_samples']]
        obs_ids = [osp.basename(x).split('.')[0].split('-')[1] for x in filenames]
        result = list(zip(result[0], obs_ids))

        batch_size = len(result)
        results.extend(result)

        prog_bar.update(batch_size)
    return results

def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def main():
    args = parse_args()

    default_scope = DefaultScope.get_instance('test', scope_name='mmpretrain')

    cfg = mmengine.Config.fromfile(args.config) #mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = DataParallel(model, device_ids=cfg.gpu_ids)
        if not model.device_ids:
            assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                'To test with CPU, please confirm your mmcv version ' \
                'is not lower than v1.4.4'

    outputs = single_gpu_test(model, data_loader)

    results = defaultdict(list)
    for result, obs_id in outputs:
        results[obs_id].append(result)

    if not args.no_scores:
        with open(args.out + '.scores.csv', 'w') as f2:
            for obs_id, result in results.items():
                avg_feats = torch.mean(torch.stack(result, dim=0), dim=0, keepdim=True)
                scores = model.module.head(avg_feats)
                f2.write(f'{obs_id}')
                for s in scores:
                    f2.write(f',{s}')
                f2.write('\n')
    
    dropped = 0
    total = 0
    with open(args.out, 'w') as f:
        f.write('observationID,class_id\n')
        for obs_id, result in results.items():
            avg_feats = torch.mean(torch.stack(result, dim=0), dim=0, keepdim=True)
            scores = model.module.head(avg_feats)
            scores = scores.detach().cpu().numpy()
            class_id = np.argmax(scores)
            if args.threshold:
                max_score = float(torch.max(torch.softmax(torch.from_numpy(scores), dim=0)))
                if max_score < args.threshold:
                    class_id = -1
                    dropped += 1
            total += 1
            f.write(f'{obs_id},{float(class_id):.1f}\n')

    print(f'dropped {dropped} out of {total}')


if __name__ == '__main__':
    main()
