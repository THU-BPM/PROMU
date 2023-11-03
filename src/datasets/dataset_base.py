from torch.utils.data import Dataset
from PIL import Image
import io
import av
import torch
import numpy as np
import lmdb
import random
import decord
from src.datasets.data_utils import (
    ImageResize, ImagePad, image_to_tensor)
from src.utils.load_save import LOGGER

decord.bridge.set_bridge("torch")


class PromuBaseDataset(Dataset):
    """
    datalist: list(dicts)  # lightly pre-processed
        {
        "type": "image",
        "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
        "text": "A plate of food and a beverage are on a table.",
        ...
        }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    """

    def __init__(self, datalist, tokenizer, img_lmdb_dir, img_db_type='rawimage',
                  max_img_size=-1, max_txt_len=20):
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_img_size = max_img_size
        self.img_resize = ImageResize(
            max_img_size,
            "bilinear")  # longer side will be resized to 1000
        self.img_pad = ImagePad(
            max_img_size, max_img_size)  # pad to 1000 * 1000

        self.img_db_type = img_db_type

        assert img_db_type in ['rawimage'], "Invalid type for img_db_type, expected {'rawimage'}, found {}.".format(img_db_type)

        self.img_db_dir = img_lmdb_dir

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        raise NotImplementedError

   

    @classmethod
    def _is_extreme_aspect_ratio(cls, tensor, max_ratio=5.):
        """ find extreme aspect ratio, where longer side / shorter side > max_ratio
        Args:
            tensor: (*, H, W)
            max_ratio: float, max ratio (>1).
        """
        h, w = tensor.shape[-2:]
        return h / float(w) > max_ratio or h / float(w) < 1 / max_ratio




def img_collate(imgs):
    """
    Args:
        imgs:

    Returns:
        torch.tensor, (B, 3, H, W)
    """
    w = imgs[0].width
    h = imgs[0].height
    tensor = torch.zeros(
        (len(imgs), 3, h, w), dtype=torch.uint8).contiguous()
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        # (H, W, 3) --> (3, H, W)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor
