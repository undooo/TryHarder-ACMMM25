from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import torch
from dataset import make_dataloader
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import os

def custom_collate_fn(batch):
    # 解包批次数据
    src_imgs = [item[0] for item in batch]  # 保持为 PIL.Image.Image 列表
    mask_imgs = [item[1] for item in batch]  # 保持为 PIL.Image.Image 列表
    text_prompts = [item[2] for item in batch]  # 文本提示列表
    src_paths = [item[3] for item in batch]  # 路径列表
    mask_paths = [item[4] for item in batch]  # 路径列表
    
    # 返回打包后的数据
    return src_imgs, mask_imgs, text_prompts, src_paths, mask_paths

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

dataset_path = '/zhaoyujian/Dataset/CC-ReID/PRCC/rgb/train'
mask_path = '/zhaoyujian/Dataset/CC-ReID/PRCC_Mask/rgb/train'

train_loader = make_dataloader()
train_loader = DataLoader(train_loader.dataset, 
                          batch_size=train_loader.batch_size, 
                          shuffle=True, collate_fn=custom_collate_fn)
height = 768
width = 256

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", 
    torch_dtype=torch.float16,
    safety_checker=None
    ).to("cuda")


epochs = 10
for epoch in range(epochs):
    save_dir = f'./output'
    os.makedirs(save_dir, exist_ok=True)
    for step, (src_imgs, mask_imgs, text_prom, src_path, mask_path) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        batch_size = len(src_imgs)
        for i in range(batch_size):
            image = Image.open(src_path[i]).convert('RGB')
            image = image.resize((width, height))
            mask = Image.open(mask_path[i]).convert('L')
            mask = mask.resize((width, height))
            
            save_path = os.path.join(save_dir, src_path[i][src_path[i].find('train'):])
            if os.path.exists(save_path):
                continue
            # image = Image.open(src_path[i])
            # image = image.resize((width, height))
            # image = np.array(image, dtype=np.float32)
            # image = image.swapaxes(0, -1).swapaxes(1, -1)
            # image /= 255.
            # # print(src_path[i])
            # # print(mask_path[i])
            # mask = mask = np.array(Image.open(mask_path[i]).resize((width, height)))
            # mask = np.expand_dims(mask, axis=-1)
            # mask = (mask>0) *255.
            # mask = mask.swapaxes(0, -1).swapaxes(1, -1)
            # mask /= 255.
            
            gen_image = pipe(prompt='change the outfit of the person', 
                     image=image, 
                     mask_image=mask, 
                     height=height, width=width, num_inference_steps=50).images[0]
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gen_image.save(save_path)
            print(f"Saved generated image to {save_path}")