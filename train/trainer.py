import os
import yaml
import random
from pathlib import Path

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.renderer import name2renderer
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn


class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 2.0e-4,
            "lr_step": 100000,
            "lr_rate": 0.5,
        },
        "img_wh": [800, 800],
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "novel_view_interval": 10000,
        "worker_num": 8,
        "random_seed": 6033,
        "fp16": False,
    }

    def _init_dataset(self, train=True):
        if train:
            self.train_set = name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
            self.train_set = DataLoader(self.train_set, 1, True, num_workers=self.cfg['worker_num'],
                                        collate_fn=dummy_collate_fn)
            print(f'train set len {len(self.train_set)}')
            self.val_set_list, self.val_set_names = [], []
            dataset_dir = self.cfg['dataset_dir']
            
            for val_set_cfg in self.cfg['val_set_list']:
                name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
                val_set = name2dataset[val_type](val_cfg, False, dataset_dir=dataset_dir)
                val_set = DataLoader(val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
                self.val_set_list.append(val_set)
                self.val_set_names.append(name)
                print(f'{name} val set len {len(val_set)}')
        else:
            test_set_cfg = {}
            name, test_type, test_cfg = test_set_cfg['name'], test_set_cfg['type'], test_set_cfg['cfg']
            test_set = name2dataset[test_type](test_cfg, False, dataset_dir=dataset_dir)
            test_set = DataLoader(test_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
            self.test_set_list.append(test_set)
            self.test_set_names.append(name)
            print(f'{name} test set len {len(test_set)}')
            
    def _init_network(self):
        self.network = name2renderer[self.cfg['network']](self.cfg, self.cfg['fp16']).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(self.optimizer, self.network)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['fp16'])

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name = cfg['name']
        self.model_dir = os.path.join('outputs/model', cfg['name'])
        if not os.path.exists(self.model_dir): 
            Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        cfg_path = os.path.join(self.model_dir, 'config.yaml')
        with open(os.path.join(self.model_dir, 'config.yaml'), 'w') as f: 
            yaml.dump(self.cfg, f)
        print(f"Save {cfg_path}.")
        
        self.pth_fn = os.path.join(self.model_dir, 'model.pth')
        self.best_pth_fn = os.path.join(self.model_dir, 'model_best.pth')

    def run(self):
        self._init_dataset() # ?? is thif?
        self._init_network()
        self._init_logger()

        best_para, start_step = self._load_model()
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], bar_format='{r_bar}')
        pbar.update(start_step)

        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            # if (step + 1) % self.cfg['novel_view_interval'] == 0:
            #     render_data = train_data.copy()
            #     render_data["render"] = True

            #     self.train_network(render_data)

            log_info = {}
            # raymarching: update grid every 16 steps
            if self.cfg['network']=='shape' and self.train_network.raymarching and step % self.train_network.update_extra_interval == 0:
                loss = self.train_network.update_extra_state()
            else:
                loss = None
                
            outputs = self.train_network(train_data)
            for loss_fn in self.train_losses:
                loss_results = loss_fn(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v
            
            loss = 0 if loss is None else loss
            for k, v in log_info.items():
                if k.startswith('loss'):
                    loss = loss + torch.mean(v)

            # loss.backward()
            # self.optimizer.step()
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')
            
            # raymarching adaptive_num_rays
            if self.cfg['network']=='shape' and self.train_network.raymarching and self.train_network.adaptive_num_rays:
                self.train_network.num_rays = int(
                    round((self.train_network.num_points / outputs["num_points"]) * self.train_network.num_rays)
                )
                
            # val
            if (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the final val set to select model!
                    val_para = val_para_cur

                if val_para > best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para = val_para
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                save_fn = None
                self._save_model(step + 1, best_para, save_fn=save_fn)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()

    def test(self):
        dataset_dir = self.cfg['dataset_dir']
        test_type = self.cfg['val_set_list'][0]['type']
        test_cfg = self.cfg['val_set_list'][0]['cfg']
        test_cfg['database_name'] = test_cfg['database_name'] + '_nvs'
        test_set = name2dataset[test_type](test_cfg, False, dataset_dir=dataset_dir, test=True)
        test_set = DataLoader(test_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
        
        print(f'test set len {len(test_set)}')
        
        self._init_network()
        best_para, start_step = self._load_model()

        self.network._init_dataset(train=False)
        self.network.cfg['test_downsample_ratio'] = False
        torch.cuda.empty_cache()
        val_results = {}
        val_para = 0
        step = 0
        
        eval_results, key_metric_val = self.val_evaluator(
            self.network, self.val_losses + self.val_metrics, test_set, step,
            self.model_name, val_set_name='test')

        psnr = np.mean(eval_results['psnr'])
        ssim = np.mean(eval_results['ssim'])
        print(psnr)
        print(ssim)

    def _load_model(self, strict=False):
        best_para, start_step = 0, 0
        if os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
