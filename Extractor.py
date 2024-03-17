
import os
import math
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms 
from PIL import Image
try:
    from diffusers.utils import  randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
import random
from tqdm import tqdm

from sdm_utils import tokenize_prompt, attack_model, preprocess, identity_loss
from PGD import L2PGDAttack as L2Attack


mask_tensor = None
global_steps = 0
transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512)
])


class attack_noise_model(nn.Module):
    def __init__(self, net1, net2, ori_latents=None):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.fn = nn.MSELoss(reduction="mean")
        self.ori_latents = ori_latents

    def forward(self, src_latents, current_t = None):
        latents =self.ori_latents * mask_tensor + src_latents * (1-mask_tensor)
        if current_t is None:
            t = random.randint(1,999)
        else:
            t = current_t
        self.net1.latents = latents
        loss1 = self.net1.forward_test(token_possibility=None, noise=None, t=t, source=True, latents=latents)

        self.net2.latents = latents
        loss2 =self.net2.forward_test(token_possibility=None, noise=None, t=t, source=True, latents=latents)
        loss = loss1 - loss2
        loss = loss 
        global global_steps 
        global_steps += 1
        if global_steps%100 == 0:
            print(t, torch.norm(latents).item(), torch.max(latents).item(), torch.min(latents).item(), loss.item())
        return loss


def main(
        training_mode = 'db_prior',
        num_imgs = 10,
        per_step = 50,
        src_dataset_mode = 0,
        version = "1.4",
        epsilon = 70.0,
        kernel_size = 16,
        steps = 1000, 
        adaptive=False,
        alpha= 2.0
):
    global mask_tensor
    checkpoint = num_imgs * per_step

    if src_dataset_mode == 0:
        data_src = "wikiart_vangogh"
        data_type = "style"
    elif src_dataset_mode == 1:
        data_src = "object_dog"
        data_type = "dog"
    
    assert version == "1.4"
    
    training_parameters = f"{num_imgs}_{per_step}_v{version}"
    src_style_dirs = f"{training_mode}/{data_src}/{training_parameters}"

    low_pass_rate = 2
    low_pass = True
    clip_min, clip_max = -100.0, 100.0

    
    if training_mode =='db_prior':
        given_prompt = f"sks {data_type}"
    else:
        given_prompt = f"a figure"
    
    atk_type = f"{low_pass}_{low_pass_rate}{kernel_size}_{epsilon}{adaptive}_{alpha}_{steps}"
    dir_name = f"Recovered_Samples"
    
    for style_name in os.listdir(src_style_dirs):
        if style_name == '.ipynb_checkpoints':
            continue
        model_id = os.path.join(src_style_dirs, style_name)
        type_list = ['membership', 'hold_out']
        saving_dirname = f"{dir_name}/{src_style_dirs}/{atk_type}/{style_name}/{type_list[0]}"
        if os.path.exists(saving_dirname):
            print(f"Existing {saving_dirname}. Skipping")
            continue      
        unet = UNet2DConditionModel.from_pretrained(f"{model_id}/checkpoint-{checkpoint}/unet", torch_dtype=torch.float16).to("cuda")
        text_encoder = CLIPTextModel.from_pretrained(f"{model_id}/checkpoint-{checkpoint}/text_encoder", torch_dtype=torch.float16).to("cuda")
        pip = StableDiffusionImg2ImgPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', safety_checker=None, unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16).to("cuda")  
        ori_pip = StableDiffusionImg2ImgPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16).to("cuda")
        device = pip._execution_device
        try:
            prompt_embeds_reference = pip._encode_prompt(
                given_prompt,
                device,
                1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
            )
        except TypeError:
            prompt_embeds_reference = pip.encode_prompt(
                given_prompt,
                device,
                1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
            )[0]
            print(prompt_embeds_reference)
  
        for class_id in tqdm(range(num_imgs)):
            for type_name in type_list:
                
                data_dirname = f"./datasets/{data_src}/{num_imgs}/{style_name}/{type_name}"
                if not os.path.exists(data_dirname):
                    os.makedirs(data_dirname)
                img_path = os.path.join(data_dirname, os.listdir(data_dirname)[class_id])
                saving_dirname = f"{dir_name}/{src_style_dirs}/{atk_type}/{style_name}/{type_name}"
        
                if not os.path.exists(saving_dirname):
                    os.makedirs(saving_dirname)

                init_image = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    init_image.save(f"{saving_dirname}/{os.listdir(data_dirname)[class_id]}_src_img.png")
                
                init_image = transform(init_image)
                ori_init_image = init_image
                # lowpass_image = init_image.filter(ImageFilter.GaussianBlur(radius=low_pass_rate))

                if low_pass:
                    with torch.no_grad():
                        ori_latents = pip.vae.encode(preprocess(init_image).half().cuda()).latent_dist.sample()* pip.vae.config.scaling_factor
                        current_latents = ori_latents.clone()
                        mask_tensor = torch.zeros([1, 4, 64, 64]).to(torch.float16).cuda()
                        per_block = 64//kernel_size
                        passing_kernel = per_block//low_pass_rate
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                mask_tensor[:, :, i*per_block:i*per_block+per_block - passing_kernel:, j*per_block:j*per_block+per_block - passing_kernel] += 1
                                mask_tensor[:, :, i*per_block+passing_kernel:i*per_block+per_block:, j*per_block+passing_kernel:j*per_block+per_block] += 1
                        current_latents *= mask_tensor
                        blurred_latents = current_latents
                        lowpass_img = pip.decode_latents(blurred_latents)
                        lowpass_image = pip.numpy_to_pil(lowpass_img)[0]
                        init_image = lowpass_image
                        lowpass_image.save(f"{saving_dirname}/low_pass_{os.listdir(data_dirname)[class_id]}.png")

                latents =  pip.vae.encode(preprocess(init_image).half().cuda()).latent_dist.sample()* pip.vae.config.scaling_factor
                print(f"ori latent dist {torch.norm(latents - ori_latents)}, max and min ori is {torch.max(ori_latents), torch.min(ori_latents)}")
                if adaptive:
                    epsilon = torch.norm(latents - ori_latents).item()
                _, _ , input_ids, ori_embedding = tokenize_prompt(pip, "a figure", pip._execution_device, randn_init=True)
                net = attack_model(pip, latents=latents, input_ids=input_ids, mode=1, prompt_embeds_reference=prompt_embeds_reference, token_possibility=None, possibility_embedding=None)
                given_prompt2 = given_prompt

                try:
                    prompt_embeds_reference_2 = ori_pip._encode_prompt(
                        given_prompt2,
                        device,
                        1,
                        do_classifier_free_guidance=False,
                        negative_prompt=None,
                    )
                except TypeError:
                    prompt_embeds_reference_2 = ori_pip.encode_prompt(
                        given_prompt2,
                        device,
                        1,
                        do_classifier_free_guidance=False,
                        negative_prompt=None,
                    )[0]
                net2 = attack_model(ori_pip, latents=latents, input_ids=input_ids, mode=1, prompt_embeds_reference=prompt_embeds_reference_2, token_possibility=None, possibility_embedding=None)

                noise = randn_tensor(net.latents.shape, generator=None, device=net.pip._execution_device, dtype=net.prompt_embeds_reference.dtype) 

                with torch.no_grad():
                    for i in range(10):
                        t = i * 100 +1
                        net.latents = latents
                        loss_value = net.forward_test(token_possibility=None, noise=noise, t=t, source=True)
                        net2.latents = latents
                        loss_value_2 = net2.forward_test(token_possibility=None, noise=noise, t=t, source=True)
                        print(t, loss_value.item(), loss_value_2.item())
                
                noise_net = attack_noise_model(net, net2, ori_latents)           
                fn = identity_loss()
                noise_net_attack = L2Attack(noise_net, fn, epsilon, steps, eps_iter=alpha, clip_min=clip_min, clip_max=clip_max, targeted=True, rand_init=False)
                
                attacked_latents = noise_net_attack.perturb(latents, torch.zeros(1).cuda())
                attacked_latents = ori_latents * mask_tensor + attacked_latents * (1-mask_tensor)
                noise = randn_tensor(net.latents.shape, generator=None, device=net.pip._execution_device, dtype=net.prompt_embeds_reference.dtype) 
                with torch.no_grad():
                    image = pip.numpy_to_pil(pip.decode_latents(attacked_latents))
                    image[0].save(f"{saving_dirname}/{os.listdir(data_dirname)[class_id]}.png")
             
if __name__ == '__main__':
    main(src_dataset_mode = 0, num_imgs = 10)
    main(src_dataset_mode = 1, num_imgs = 2)
  