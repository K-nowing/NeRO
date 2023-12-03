from torch.utils.data import Dataset
import numpy as np

from dataset.database import get_database_split, parse_database_name



class DummyDataset(Dataset):
    default_cfg = {
        'database_name': '',
    }

    def __init__(self, cfg, is_train, dataset_dir=None, test=False):
        self.cfg = {**self.default_cfg, **cfg}
        if not is_train:
            if self.cfg['database_name'].split('/')[0] == 'nerf':
                self.test_num = 200
                self.test_num = 1
            else:
                if test:
                    database = parse_database_name(self.cfg['database_name'], dataset_dir)
                    self.test_num = len(database.get_img_ids())
                else:
                    database = parse_database_name(self.cfg['database_name'], dataset_dir)
                    train_ids, test_ids = get_database_split(database, 'validation')
                    self.train_num = len(train_ids)
                    self.test_num = len(test_ids)
                
            
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            return {}
        else:
            return {'index': index}

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return self.test_num
