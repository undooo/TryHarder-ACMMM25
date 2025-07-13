from diffusers import StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import make_dataloader

# ==========【1. 配置参数】==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = '/zhaoyujian/Dataset/CC-ReID/last'
output_path = '/zhaoyujian/Dataset/CC-ReID/last_inpaint'

height, width = 768, 256  # 生成图像尺寸
batch_size = 64  # 可调整 batch 大小
epochs = 10  # 训练轮数

# ==========【2. 定义数据集的 collate_fn】==========
def custom_collate_fn(batch):
    """
    重新组织 DataLoader 输出的 batch 数据
    """
    src_imgs = [item[0] for item in batch]  # 原图
    mask_imgs = [item[1] for item in batch]  # Mask
    text_prompts = [item[2] for item in batch]  # 文本提示
    src_paths = [item[3] for item in batch]  # 原图路径
    mask_paths = [item[4] for item in batch]  # Mask路径
    return src_imgs, mask_imgs, text_prompts, src_paths, mask_paths

# ==========【3. 加载数据】==========
train_loader = make_dataloader()
train_loader = DataLoader(train_loader.dataset, 
                          batch_size=train_loader.batch_size, 
                          shuffle=True, collate_fn=custom_collate_fn)

# ==========【4. 加载 Stable Diffusion Inpainting 模型】==========
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", 
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# ==========【5. 定义批量推理函数】==========
def batch_inpaint(pipe, images, masks, prompts, height, width):
    """
    批量推理的封装函数
    :param pipe: Stable Diffusion Inpaint Pipeline
    :param images: List[PIL.Image] 输入图像
    :param masks: List[PIL.Image] 遮罩图像
    :param prompts: List[str] 文本提示
    :param height: int 目标高度
    :param width: int 目标宽度
    :return: 生成的图像列表
    """
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])

    # 预处理图片 & Mask
    image_tensors = torch.stack([transform(img) for img in images])  # [batch, 3, H, W]
    mask_tensors = torch.stack([transform(mask) for mask in masks])  # [batch, 1, H, W]

    # 归一化到 [-1, 1]
    image_tensors = image_tensors * 2 - 1  
    # mask_tensors = mask_tensors.expand(-1, 3, -1, -1)  # Mask 扩展到 3 通道

    # 传入 pipeline 进行批量推理
    with torch.no_grad():
        output = pipe(
            prompt=prompts,  # 支持 batch
            image=image_tensors.to(device, dtype=pipe.unet.dtype),
            mask_image=mask_tensors.to(device, dtype=pipe.unet.dtype),
            height=height,
            width=width,
            num_inference_steps=50
        ).images
    
    return output  # 返回生成的图片列表

# ==========【6. 训练循环】==========
for epoch in range(epochs):
    save_dir = os.path.join(output_path, f'epoch{epoch+1}')
    os.makedirs(save_dir, exist_ok=True)

    for step, (src_imgs, mask_imgs, text_prompts, src_paths, mask_paths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        batch_src, batch_mask, batch_prompt, batch_save_paths = [], [], [], []
        batch_ori_size = []

        for i in range(len(src_imgs)):
            image = Image.open(src_paths[i]).convert('RGB').resize((width, height))
            mask = Image.open(mask_paths[i]).convert('L').resize((width, height))
            
            save_path = os.path.join(save_dir, os.path.relpath(src_paths[i], dataset_path))
            if os.path.exists(save_path):
                continue  # 跳过已处理的文件
            
            batch_src.append(image)
            batch_mask.append(mask)
            batch_prompt.append("change the outfit of the person")
            batch_save_paths.append(save_path)
            batch_ori_size.append(Image.open(src_paths[i]).size)


            # **【当 batch 满了，开始推理】**
            if len(batch_src) == batch_size:
                gen_images = batch_inpaint(pipe, batch_src, batch_mask, batch_prompt, height, width)

                # **保存生成的图像**
                for img, path, original_size in zip(gen_images, batch_save_paths, batch_ori_size):
                    img = img.resize(original_size, Image.LANCZOS)  # 还原尺寸
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    img.save(path)

                # **清空 batch**
                batch_src, batch_mask, batch_prompt, batch_save_paths = [], [], [], []
                batch_ori_size = []

        # **处理剩余未满 batch 的数据**
        if batch_src:
            gen_images = batch_inpaint(pipe, batch_src, batch_mask, batch_prompt, height, width)
            for img, path in zip(gen_images, batch_save_paths):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path)
