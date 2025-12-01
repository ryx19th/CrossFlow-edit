from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from scipy.signal import convolve2d
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import time
from tqdm import tqdm
import json
import pickle
import io
import cv2

import libs.clip
import bisect

from ipdb import set_trace as st


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


# class MSCOCODatabase(Dataset):
#     def __init__(self, root, annFile, size=None):
#         from pycocotools.coco import COCO
#         self.root = root
#         self.height = self.width = size

#         self.coco = COCO(annFile)
#         self.keys = list(sorted(self.coco.imgs.keys()))

#     def _load_image(self, key: int):
#         path = self.coco.loadImgs(key)[0]["file_name"]
#         return Image.open(os.path.join(self.root, path)).convert("RGB")

#     def _load_target(self, key: int):
#         return self.coco.loadAnns(self.coco.getAnnIds(key))

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, index):
#         key = self.keys[index]
#         image = self._load_image(key)
#         image = np.array(image).astype(np.uint8)
#         image = center_crop(self.width, self.height, image).astype(np.float32)
#         image = (image / 127.5 - 1.0).astype(np.float32)
#         image = einops.rearrange(image, 'h w c -> c h w')

#         anns = self._load_target(key)
#         target = []
#         for ann in anns:
#             target.append(ann['caption'])

#         return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


# class MSCOCOFeatureDataset(Dataset):
#     # the image features are got through sample
#     def __init__(self, root, need_squeeze=False, full_feature=False, fix_test_order=False):
#         self.root = root
#         self.num_data, self.n_captions = get_feature_dir_info(root)
#         self.need_squeeze = need_squeeze
#         self.full_feature = full_feature
#         self.fix_test_order = fix_test_order

#     def __len__(self):
#         return self.num_data

#     def __getitem__(self, index):
#         if self.full_feature:
#             z = np.load(os.path.join(self.root, f'{index}.npy'))

#             if self.fix_test_order:
#                 k = self.n_captions[index] - 1
#             else:
#                 k = random.randint(0, self.n_captions[index] - 1)

#             test_item = np.load(os.path.join(self.root, f'{index}_{k}.npy'), allow_pickle=True).item()
#             token_embedding = test_item['token_embedding']
#             token_mask = test_item['token_mask']
#             token = test_item['token']
#             caption = test_item['promt']
#             return z, token_embedding, token_mask, token, caption
#         else:
#             z = np.load(os.path.join(self.root, f'{index}.npy'))
#             k = random.randint(0, self.n_captions[index] - 1)
#             c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
#             if self.need_squeeze:
#                 return z, c.squeeze()
#             else:
#                 return z, c


class ImageDataset(Dataset):
    def __init__(self, root, resolution, llm, edit_mode=False, prompt_mode='output', naive_mode=None):
        super().__init__()
        json_path = os.path.join(root,'img_text_pair.jsonl')
        self.img_root = os.path.join(root,'imgs')
        self.resolution = resolution
        self.llm = llm
        self.file_list = []
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                self.file_list.append(json.loads(line))
        self.edit_mode = edit_mode
        self.prompt_mode = prompt_mode
        assert self.prompt_mode in ['output', 'dual', 'instruction']
        self.naive_mode = naive_mode
        assert self.naive_mode in [None, 'zoom_in', 'zoom_in_out', 'rotate', 'hole', 'hole_latent', 'margin', 'hole_margin', ]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data_item = self.file_list[idx]

        img_path = os.path.join(self.img_root, data_item['img'])
        img, pil_img = self.load_process_image(img_path)

        if self.edit_mode:
            src_img_path = os.path.join(self.img_root, data_item['img_src'])
            img_src, pil_img_src = self.load_process_image(src_img_path)

        if self.prompt_mode == 'output':
            prompt = data_item['prompt']
        elif self.prompt_mode == 'dual':
            prompt = [data_item['prompt'], data_item['prompt_src']]

            # if prompt[1] == '':
            #     prompt[1] = 'A photo'
            #     prompt[0] = 'A photo that ' + prompt[0]
            #     # Just for numerical stablity, but still not sure if really here problem.
            #     # And still not good data for many src prompt, better to use instrcution
            #     # And even better to directly use gpt-image dataset etc. and so on ........ (any more extended data quality) 

        elif self.prompt_mode == 'instruction':
            prompt = data_item['prompt_inst']

        # or do src to aug (at p)
        # also for natural logic
        if self.naive_mode == 'zoom_in':
            img_zoom = self.zoom_in_img(pil_img)
            img, img_src = img_zoom, img
            prompt = 'zoom in'

        elif self.naive_mode == 'zoom_in_out':
            img_zoom = self.zoom_in_img(pil_img)
            p = random.random()
            if p < 0.5:
                img, img_src = img_zoom, img
                prompt = 'zoom in'
            else:
                img_src = img_zoom
                prompt = 'zoom out'
            # diff = np.abs(img - img_src).max()
            # if diff < 1e-2:
            #     print(f"[DEBUG] idx={idx}, path={img_path}, prompt={prompt}, max_diff={diff}")

        elif self.naive_mode == 'rotate':
            p = random.random()
            if p < 0.5:
                img_rot = self.rotate_img(pil_img, 'left')
                img, img_src = img_rot, img
                prompt = 'rotate left'
            else:
                img_rot = self.rotate_img(pil_img, 'right')
                img, img_src = img_rot, img
                prompt = 'rotate right'

        elif self.naive_mode == 'hole':
            img_hole = self.hole_margin_img(img, mode='hole')
            img, img_src = img_hole, img
            prompt = 'an image with a hole in the center'

        elif self.naive_mode == 'hole_latent':
            # st()
            img_src = img

        elif self.naive_mode == 'margin':
            img_mar = self.hole_margin_img(img, mode='margin')
            img, img_src = img_mar, img
            prompt = 'an image with margins missing'

        elif self.naive_mode == 'hole_margin':
            p = random.random()
            if p < 0.5:
                img_hole = self.hole_margin_img(img, mode='hole')
                img, img_src = img_hole, img
                prompt = 'an image with a hole in the center'
            else:
                img_mar = self.hole_margin_img(img, mode='margin')
                img, img_src = img_mar, img
                prompt = 'an image with margins missing'

        if self.edit_mode:
            return img, prompt, img_src
        else:
            return img, prompt

    def load_process_image(self, img_path):
        pil_image = Image.open(img_path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        pil_image = pil_image.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        img = np.array(pil_image).astype(np.float32)
        img = img / 127.5 - 1.0
        img = einops.rearrange(img, 'h w c -> c h w')
        return img, pil_image

    def zoom_in_img(self, pil_img):
        # h, w = pil_img.size
        # assert h == self.resolution and w == self.resolution
        left = self.resolution // 4
        top = self.resolution // 4
        right = self.resolution * 3 // 4
        bottom = self.resolution * 3 // 4
        pil_img = pil_img.crop((left, top, right, bottom))
        pil_img = pil_img.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        img = np.array(pil_img).astype(np.float32)
        img = img / 127.5 - 1.0
        img = einops.rearrange(img, 'h w c -> c h w')
        return img

    def rotate_img(self, pil_img, direction):
        if direction == 'left':
            pil_img = pil_img.transpose(Image.Transpose.ROTATE_90)
        elif direction == 'right':
            pil_img = pil_img.transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError("Direction must be 'left' or 'right'.")
        img = np.array(pil_img).astype(np.float32)
        img = img / 127.5 - 1.0
        img = einops.rearrange(img, 'h w c -> c h w')
        return img

    def hole_margin_img(self, img, mode):
        assert mode in ['hole', 'margin']
        if mode == 'hole':
            img_hole = img.copy()
            start_idx = self.resolution // 4
            end_idx = self.resolution * 3 // 4
            img_hole[:, start_idx:end_idx, start_idx:end_idx] = -1.0 # enc z -3 +3 but here not latent just tensor! so still rgb and -1 +1 for sure! # 
            return img_hole
        elif mode == 'margin':
            img_mar = img.copy()
            start_idx = self.resolution // 4
            end_idx = self.resolution * 3 // 4
            img_mar[:, :start_idx, :] = -1.0
            img_mar[:, end_idx:, :] = -1.0
            img_mar[:, :, :start_idx] = -1.0
            img_mar[:, :, end_idx:] = -1.0
            return img_mar


class ImageFullDataset(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, train_path, val_path, resolution, llm, cfg=False, p_uncond=None, fix_test_order=False, edit_mode=False, prompt_mode='output', naive_mode=None):
        super().__init__()
        print('Prepare dataset...')
        self.resolution = resolution

        self.train = ImageDataset(train_path, resolution=resolution, llm=llm, edit_mode=edit_mode, prompt_mode=prompt_mode, naive_mode=naive_mode)
        self.test = ImageDataset(val_path, resolution=resolution, llm=llm, edit_mode=edit_mode, prompt_mode=prompt_mode, naive_mode=naive_mode)
        
        print('Prepare dataset ok')

        # self.empty_context = np.load(os.path.join(val_path, 'empty_context.npy'), allow_pickle=True).item()

        # assert not cfg

        # text embedding extracted by clip
        # self.prompts, self.token_embedding, self.token_mask, self.token = [], [], [], []
        # for f in sorted(os.listdir(os.path.join(val_path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
        #     vis_item = np.load(os.path.join(val_path, 'run_vis', f), allow_pickle=True).item()
        #     self.prompts.append(vis_item['promt'])
        #     self.token_embedding.append(vis_item['token_embedding'])
        #     self.token_mask.append(vis_item['token_mask'])
        #     self.token.append(vis_item['token'])
        # self.token_embedding = np.array(self.token_embedding)
        # self.token_mask = np.array(self.token_mask)
        # self.token = np.array(self.token)

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32

    # @property
    # def fid_stat(self):
    #     return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'ImageDataset':
        return ImageFullDataset(**kwargs)
    else:
        raise NotImplementedError(name)
