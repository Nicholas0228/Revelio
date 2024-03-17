from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import torch
import numpy as np
import PIL
try:
    from diffusers.utils import PIL_INTERPOLATION, randn_tensor
except ImportError:
    from diffusers.utils import PIL_INTERPOLATION
    from diffusers.utils.torch_utils import randn_tensor
# from ddnm_sdm import *
import torch.nn as nn
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling
from typing import List, Optional


def tokenize_prompt(
    pip,
    prompt,
    device,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    randn_init = False
):
    if prompt_embeds is None:
        text_inputs = pip.tokenizer(
            prompt,
            padding="max_length",
            max_length=pip.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(pip.text_encoder.config, "use_attention_mask") and pip.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        if randn_init:
            text_input_ids = torch.randint(low=0, high=49408, size=text_input_ids.shape, dtype=torch.int64)

        # print(text_input_ids)
        # (token_embedding): Embedding(49408, 768)
        # (position_embedding): Embedding(77, 768)
        hidden_states = pip.text_encoder.text_model.embeddings(input_ids=text_input_ids.to(device))
        token_embedding = pip.text_encoder.text_model.embeddings.token_embedding(text_input_ids.to(device))
        return hidden_states, attention_mask, text_input_ids.to(device), token_embedding


def get_position_embedding(pip, seq_length):
    # print(pip.text_encoder.text_model.embeddings.position_ids[:, :seq_length])
    return pip.text_encoder.text_model.embeddings.position_embedding(pip.text_encoder.text_model.embeddings.position_ids[:, :seq_length])


def text_encoder_processing(pip,
    text_encoder,
    text_possibility = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    possibility_embedding=None,
):
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else text_encoder.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_encoder.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_encoder.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    # with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
    #     text_embeddings = possibility_embedding(text_possibility)
    text_embeddings = possibility_embedding(text_possibility)
    hidden_states = text_embeddings + get_position_embedding(pip, input_ids.shape[-1])


    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = text_encoder._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )

    encoder_outputs = text_encoder.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.final_layer_norm(last_hidden_state)
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def process_prompt(pip, hidden_states, attention_mask=None, device=None, input_ids=None, possibility_embedding=None):
    # prompt_embeds = pip.text_encoder(
    #     text_input_ids,
    #     attention_mask=attention_mask,
    # )

    prompt_embeds = text_encoder_processing(pip, pip.text_encoder.text_model, hidden_states, input_ids, possibility_embedding=possibility_embedding)
    prompt_embeds = prompt_embeds[0]

    prompt_embeds_dtype = pip.text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
    return prompt_embeds



def return_para_from_pip(
    pip,
    original_samples: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = pip.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod



class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class attack_model(nn.Module):
    def __init__(self, pip, latents, input_ids, mode=0, prompt_embeds_reference=None, token_possibility=None, possibility_embedding=None):
        super().__init__()
        self.pip = pip
        self.latents = latents
        self.fn = nn.MSELoss(reduction="mean")
        self.input_ids = input_ids
        self.mode = mode
        self.prompt_embeds_reference  = prompt_embeds_reference 
        if token_possibility:
            self.token_possibility = token_possibility.clone()
        self.possibility_embedding = possibility_embedding

    def forward_test(self, token_possibility, noise, t, source=False, return_noise=False, latents=None):
        if latents is None:
            current_latents = self.latents
        else:
            current_latents = latents
        if noise is None:
            noise = randn_tensor(current_latents.shape, generator=None, device=self.pip._execution_device, dtype=self.prompt_embeds_reference.dtype) 
        if source:
            prompt_embeds = self.prompt_embeds_reference
        else:
            prompt_embeds = process_prompt(self.pip, token_possibility, None, self.pip._execution_device, self.input_ids, possibility_embedding=self.possibility_embedding)
        device = noise.device
        source_latents =  current_latents
        latent_timestep = torch.tensor(t).repeat(1).to(device)
        alpha1, alpha2 = return_para_from_pip(self.pip.scheduler, source_latents, latent_timestep)
        latents = alpha1 * source_latents + alpha2 * noise

        latent_model_input = latents
        noise_pred = self.pip.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
            ).sample
        if noise_pred.shape[1] == 6:
            noise_pred, _ = torch.chunk(noise_pred, 2, dim=1)
        loss = self.fn(noise_pred.float(), noise.float())
        if return_noise:
            return loss, noise_pred
        return loss



def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)

    return image

