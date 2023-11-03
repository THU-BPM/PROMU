import os
import json
import random

import torch
import spacy
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import ImageRandomSquareCrop, ImageResizeSquare
from src.datasets.dataset_base import PromuBaseDataset, img_collate

from src.datasets.randaugment import  RandomAugment

from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import numpy as np


class PromuPretrainSparseDataset(PromuBaseDataset):
    """
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict {
            "type": "image",
            "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
            "text": "A plate of food and a beverage are on a table."
            ...
            }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    vis_format: str, image or image, used to decide data loading method.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, img_db_type, txt_dir,
                crop_size=256, resize_size=288, 
                max_img_size=1000, max_txt_len=20,
                use_itm=True, is_train=True):
        super(PromuPretrainSparseDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, 
            img_db_type=img_db_type,
            max_img_size=max_img_size, 
            max_txt_len=max_txt_len)
        self.use_itm = use_itm

        self.txt_dir = txt_dir

        self.crop_size = crop_size
        self.image_random_cropper = ImageRandomSquareCrop(crop_size)

        self.resize_size = resize_size

        self.is_train = is_train

        if self.is_train:
            self.randaug = RandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        start_time = None
        end_time = None

        # fetch image
        num_retries = 10  # skip error images

        for _ in range(num_retries):
            data_sample = self.datalist[index][1][0]
            if 'text' in data_sample:
                text = data_sample["text"].strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch image
            image_path = os.path.join(self.img_db_dir, data_sample["filepath"]) 

            # read with retries
            for i in range(3):
                img_array = Image.open(image_path)
                img_array = img_array.resize((self.resize_size,self.resize_size),Image.ANTIALIAS)

                if img_array is not None:
                    break


            # Select a random image if the current image was not able to access.
            if img_array is None:
                LOGGER.info(f"Failed to load examples with image: {image_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                # square crop
                img_array = self.image_random_cropper(img_array)

                if self.randaug:
                    img_array = self.randaug(img_array)

                break
        else:
            raise RuntimeError(f"Failed to fetch image after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': 1}]

        return dict(
            img=img_array,  # (T, C, H, W)
            examples=examples,
            n_examples=len(examples),  # used to create image feature copies.
            type='image'
        )

class PretrainImageTextDataset(Dataset):
    def __init__(self, datalist, tokenizer, is_train=True, crop_size=256, resize_size=288, num_frm=4, max_txt_len=40):
        self.datalist = datalist
        self.max_txt_len = max_txt_len

        self.crop_size = crop_size
        self.resize_size = resize_size

        self.is_train = is_train

        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])     
            ])    
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        start_time = None
        end_time = None

        # fetch image
        num_retries = 10  # skip error images

        for _ in range(num_retries):
            data_sample = self.datalist[index][1][0]
            try:
                if type(data_sample['text']) == list:
                    text = random.choice(data_sample['text'])
                else:
                    text = data_sample['text']
                label = data_sample['label']
                obj_top = data_sample['obj_top']
                obj_left = data_sample['obj_left']
                obj_h = data_sample['obj_h']
                obj_w = data_sample['obj_w']
                img_path = os.path.join("../origin",data_sample['filepath'])
                img_arr = Image.open(img_path).convert('RGB')  
                img_arr = self.transform(img_arr)
                img_arr = np.asarray(img_arr, dtype=np.float32).transpose(2, 0, 1)
                img_arr = torch.from_numpy(img_arr)
                obj_path = os.path.join("../origin",data_sample['obj_path'])
                obj_arr = Image.open(img_path).convert('RGB')  
                obj_arr = self.transform(obj_arr)
                obj_arr = np.asarray(obj_arr, dtype=np.float32).transpose(2, 0, 1)
                obj_arr = torch.from_numpy(obj_arr)
            except Exception as e:
                img_arr = None

            if img_arr is not None:
                c,w,h = img_arr.shape
            # Select a random image if the current image was not able to access.
            if img_arr is None:
                LOGGER.info(f"Failed to load examples with image: {img_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:
            raise RuntimeError(f"Failed to fetch image after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': label}]

        return dict(
            img=img_arr,  # (C, H, W)
            obj = obj_arr,
            obj_top = obj_top,
            obj_left = obj_left,
            obj_h = obj_h,
            obj_w = obj_w,
            examples=examples,
            n_examples=len(examples),  # used to create image feature copies.
            type='img'
        )


class PretrainCollator(object):
    def __init__(self, tokenizer, 
                 patch_size=16,
                 cir = True,
                 coe = True,
                 max_length=20, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.cir = cir
        self.coe = coe
        self.patch_size = patch_size

    def collate_batch(self, batch):
        if isinstance(batch[0]["img"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        visual_inputs = v_collate([d["img"] for d in batch])  

        obj_inputs = v_collate([d['obj'] for d in batch]) 
        obj_inputs_top = v_collate([d['obj_top'] for d in batch]) 
        obj_inputs_left = v_collate([d['obj_left'] for d in batch]) 
        obj_inputs_w = v_collate([d['obj_w'] for d in batch]) 
        obj_inputs_h = v_collate([d['obj_h'] for d in batch]) 
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  
        # group elements data
        batch_enc = self.tokenizer.batch_encode_plus(
            [d["text_str"] for d in text_examples],
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_ids_no_mask = text_input_ids.clone()        
        text_input_mask = batch_enc.attention_mask  # (B, L)
        itm_labels = default_collate(
            [d["itm_label"] for d in text_examples])  # (B, )
        
        erase_elems = []
        for input, obj_top, obj_left, obj_w, obj_h in zip(visual_inputs.clone(),obj_inputs_top,obj_inputs_left,obj_inputs_w,obj_inputs_h):
            erase_elems.append(erase(input, patch_size=self.patch_size,obj_top = obj_top,obj_left = obj_left,obj_w = obj_w, obj_h = obj_h))
        if self.cir and self.coe:
            crop_visual_inputs_image = obj_inputs
            coe_masks = v_collate([elems[1] for elems in erase_elems])
            context_visual_inputs_image = v_collate([elems[2] for elems in erase_elems])
            cir_masks = v_collate([torch.zeros_like(elems[1]) for elems in erase_elems])
            return dict(
                visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
                obj_inputs = obj_inputs,
                crop_visual_inputs_image=crop_visual_inputs_image,
                context_visual_inputs_image=context_visual_inputs_image,
                cir_mask=cir_masks,
                crop_visual_inputs_object=crop_visual_inputs_image,
                context_visual_inputs_object=crop_visual_inputs_image,
                coe_mask=coe_masks,
                text_input_ids=text_input_ids_no_mask,
                text_input_mask=text_input_mask, # used to exclude [PAD] token
                itm_labels=itm_labels,
                n_examples_list=n_examples_list,  # used to create image feature copies.
                type=batch[0]['type']
            )
        else:
            return dict(
                visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
                text_input_ids=text_input_ids_no_mask,
                text_input_mask=text_input_mask, # used to exclude [PAD] token
                itm_labels=itm_labels,
                n_examples_list=n_examples_list,  # used to create image feature copies.
                type=batch[0]['type']
            )

def random_erase(input_img, patch_size, s_l=0.3, s_h=0.5, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    assert input_img.ndim == 3
    img_c, img_h, img_w = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        w = w - w % patch_size
        h = h - h % patch_size

        left = left - left % patch_size
        top = top - top % patch_size

        if left + w <= img_w and top + h <= img_h:
            break

    context_img = input_img.clone()
    context_img[ :, top: top + h, left: left + w] = 0

    input_img = input_img[ :, top: top + h, left: left + w]
    pad = (left, img_w - left - w, top, img_h - top - h)
    input_img = torch.nn.functional.pad(input_img, pad=pad, mode='constant', value=0.0)

    img_masks = torch.ones_like(input_img)
    img_masks[ :, top: top+h, left: left+w] = 0

    img_masks = torch.nn.functional.avg_pool2d(img_masks.float(), kernel_size=(patch_size, patch_size), stride=patch_size)
    img_masks = torch.mean(img_masks, dim=0)

    return input_img, img_masks, context_img
def erase(input_img, patch_size, top, left, w, h):
    assert input_img.ndim == 3
    img_c, img_h, img_w = input_img.shape

    context_img = input_img.clone()
    context_img[ :, top: top + h, left: left + w] = 0

    input_img = input_img[ :, top: top + h, left: left + w]
    pad = (left, img_w - left - w, top, img_h - top - h)
    input_img = torch.nn.functional.pad(input_img, pad=pad, mode='constant', value=0.0)

    img_masks = torch.ones_like(input_img)
    img_masks[ :, top: top+h, left: left+w] = 0

    img_masks = torch.nn.functional.avg_pool2d(img_masks.float(), kernel_size=(patch_size, patch_size), stride=patch_size)
    img_masks = torch.mean(img_masks, dim=0)

    return input_img, img_masks, context_img