import time

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from network.metrics import name2key_metrics
from train.train_tools import to_cuda


class ValidationEvaluator:
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        self.key_metric_name = cfg['key_metric_name']
        self.key_metric = name2key_metrics[self.key_metric_name]

    def __call__(self, model, losses, metrics, eval_dataset, step, val_set_name=None):
        model.eval()
        eval_results = {}
        begin = time.time()
        result_image_dict = dict()
        for data_i, data in enumerate(tqdm(eval_dataset)):
            data = to_cuda(data)
            data['eval'] = True
            data['step'] = step
            with torch.no_grad():
                outputs = model(data)

            for loss in losses:
                loss_results = loss(outputs, data, step, data_index=data_i)
                for k, v in loss_results.items():
                    if type(v) == torch.Tensor:
                        v = v.detach().cpu().numpy()

                    if k in eval_results:
                        eval_results[k].append(v)
                    else:
                        eval_results[k] = [v]
            metric_results, result_image = metrics(outputs, data, step, data_index=data_i)
            for k, v in metric_results.items():
                if type(v) == torch.Tensor:
                    v = v.detach().cpu().numpy()
                if k in eval_results:
                    eval_results[k].append(v)
                else:
                    eval_results[k] = [v]
            result_image_dict[data_i] = result_image

        for k, v in eval_results.items():
            eval_results[k] = np.concatenate(v, axis=0).mean().item()

        # evaluate poses
        # with torch.no_grad():
        #     eval_results.update(model.evaluation())

        key_metric_val = self.key_metric(eval_results)
        eval_results[self.key_metric_name] = key_metric_val
        print('eval cost {} s'.format(time.time() - begin))
        torch.cuda.empty_cache()
        return eval_results, key_metric_val, result_image_dict
