import os
import numpy as np
from abc import abstractmethod
from pathlib import Path
from einops import rearrange
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass

class InpaintingDataset(Dataset):
    def __init__(self, data_path, mask_path, split):
        self.data_dir = Path(data_path)
        self.mask_dir = Path(mask_path)

        self.split = split

        self.transforms = transforms.Compose([transforms.Resize((512, 512)),
                                              transforms.ToTensor(),
                                              transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')),])
        
        self.face_transforms = transforms.Compose([transforms.Resize((512, 512)),
                                                   transforms.ToTensor(),])
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        masks = sorted([f for f in self.mask_dir.iterdir() if f.suffix == '.png'])
        
        self.imgs = imgs[:2000] if split == "train" else imgs[2000:2500]
        self.masks = masks[:2000] if split == "train" else masks[2000:2500]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        mask = default_loader(self.masks[idx])

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1
        mask = torch.from_numpy(mask[..., None])
        print(mask.shape)
        print(img.shape)
        
        face = self.face_transforms(img)
        img = self.transforms(img)
        masked_image = img * (mask < 0.5)

        return {
            "jpg": img,
            "mask": mask,
            "masked_image": masked_image,
            "face": face,
        }


class PRNGMixin(object):
    """
    Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing.
    """
    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng