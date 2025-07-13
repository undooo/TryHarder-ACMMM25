import sys
sys.path.append('./')
from PIL import Image
import glob
from tqdm import tqdm
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j]:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def load_models():
    print("loading models...")
    base_path = '/root/IDM-VTON/ckpt/'
    
    unet = UNet2DConditionModel.from_pretrained(
        base_path, subfolder="unet", torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path, subfolder="tokenizer", revision=None, use_fast=False,
    )
    
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path, subfolder="tokenizer_2", revision=None, use_fast=False,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=torch.float16,
    )
    
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path, subfolder="text_encoder_2", torch_dtype=torch.float16,
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path, subfolder="image_encoder", torch_dtype=torch.float16,
    )
    
    vae = AutoencoderKL.from_pretrained(
        base_path, subfolder="vae", torch_dtype=torch.float16,
    )
    
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path, subfolder="unet_encoder", torch_dtype=torch.float16,
    )
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    
    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    
    # 将模型移至设备
    pipe.to(device)
    pipe.unet_encoder.to(device)
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    print("model loaded")
    return pipe, parsing_model, openpose_model

def process_image(human_img_path, garm_img_path, pipe, parsing_model, openpose_model, 
                  garment_desc="casual clothes", denoise_steps=30, seed=42):
    """Handle the clothing change of a single picture"""
    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    
    garm_img = Image.open(garm_img_path).convert("RGB").resize((768, 1024))
    human_img_orig = Image.open(human_img_path).convert("RGB")
    human_img = human_img_orig.resize((768, 1024))

   
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = mask.resize((768, 1024))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    args = apply_net.create_argument_parser().parse_args((
        'show', 
        './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 
        'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)    
    pose_img = pose_img[:, :, ::-1]    
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    with torch.no_grad():
       
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_desc
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                prompt = "a photo of " + garment_desc
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, list):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, list):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img, 
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]

    return images[0]


def main():
    
    dataset_name = "ltcc"

    if dataset_name == "ltcc":
        person_folder = "/root/autodl-tmp/ori-data/LTCC_ReID/continue_per5"
        cloth_folder = "/root/autodl-tmp/ori-data/LTCC_ReID/Cloth_HQ"
        output_folder = "/root/autodl-tmp/ori-data/LTCC_ReID/Cloth_Changed_PersonPer5"
    elif dataset_name == "prcc":
        person_folder = "/root/autodl-tmp/ori-data/prcc/rgb/person_hq_per5"
        cloth_folder = "/root/autodl-tmp/ori-data/LTCC_ReID/Cloth_HQ"
        output_folder = "/root/autodl-tmp/ori-data/prcc/rgb/5_N_cc"
    else:
        raise  ValueError("dataset name must be set!")


   
    os.makedirs(output_folder, exist_ok=True)
    
 
    pipe, parsing_model, openpose_model = load_models()
    

    person_images = []
    cloth_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
        person_images.extend(glob.glob(os.path.join(person_folder, ext)))
        cloth_images.extend(glob.glob(os.path.join(cloth_folder, ext)))
    person_images = sorted(person_images)
    cloth_images = sorted(cloth_images)
    
    print(f"Found {len(person_images)} person images and {len(cloth_images)} clothing images")
    total_combinations = len(person_images) * len(cloth_images)
    print(f"A total of {total_combinations} combinations need to be processed")
    
    progress_bar = tqdm(total=total_combinations, desc="Processing progress")
    for person_path in person_images:
        person_filename = os.path.basename(person_path)
        person_basename = os.path.splitext(person_filename)[0]
        name_parts = person_basename.split("_")
        
        for cloth_path in cloth_images:
            cloth_filename = os.path.basename(cloth_path)
            cloth_id = int(os.path.splitext(cloth_filename)[0])
            
            if dataset_name in ["ltcc", "prcc"]:
                new_parts = name_parts.copy()
                new_parts[1] = str(100+cloth_id)
                new_filename = "_".join(new_parts) + ".png"
            
            
            output_path = os.path.join(output_folder, new_filename)
            
            try:
                result_img = process_image(
                    person_path, 
                    cloth_path, 
                    pipe, 
                    parsing_model, 
                    openpose_model,
                    garment_desc="casual garment",
                    denoise_steps=30, 
                    seed=42
                )
                
                result_img.save(output_path)
                progress_bar.set_postfix({"current": f"{person_filename} + {cloth_filename}"})
            except Exception as e:
                print(f"An error occurred while processing {person_filename} and {cloth_filename}: {e}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    print("Processing completed! All results have been saved to", output_folder)

if __name__ == "__main__":
    main()