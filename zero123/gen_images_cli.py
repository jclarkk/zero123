import argparse
import math
import numpy as np
import os
import time
import torch
from PIL import Image
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from lovely_numpy import lo
from omegaconf import OmegaConf
from rich import print
from torch import autocast
from torchvision import transforms
from transformers import AutoFeatureExtractor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_and_preprocess, create_carvekit_interface, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='3D Image Generation')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument('--preprocess', type=bool, default=True, help='Whether to preprocess the image.')
    parser.add_argument('--polar_angle', type=str, default='0.0', help='Polar angle (vertical rotation in degrees).')
    parser.add_argument('--azimuth_angle', type=str, default='0.0',
                        help='Azimuth angle (horizontal rotation in degrees).')
    parser.add_argument('--zoom', type=float, default=0.0, help='Zoom (relative distance from center).')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples to generate.')
    parser.add_argument('--scale', type=float, default=3.0, help='Diffusion guidance scale.')
    parser.add_argument('--ddim_steps', type=int, default=75, help='Number of diffusion inference steps.')
    args = parser.parse_args()
    return args


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


def init_models(config, ckpt, device):
    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    # Reduce NSFW false positives.
    # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
    # models['nsfw'].concept_embeds_weights:
    # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
    #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
    # models['nsfw'].special_care_embeds_weights:
    # [0.1950, 0.2000, 0.2200].
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    models['nsfw'].concept_embeds_weights *= 1.07
    models['nsfw'].special_care_embeds_weights *= 1.07

    return models


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main_cli(models, device, x='0.0', y='0.0', z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256, output_folder=None):
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

    input_im = process_image(device, models, preprocess, raw_im, h, w)

    if ',' in x and ',' in y:
        raise EnvironmentError("Cannot use arrays in azimuth and polar, use only one.")

    os.makedirs(output_folder, exist_ok=True)

    index = 0
    if ',' in x:
        x_list = x.split(',')
        prev_x = 0
        for current_x in x_list:
            current_x = float(current_x)
            x_delta = current_x if index == 0 else current_x - prev_x
            print(f'Current: {current_x} delta: {x_delta}')
            output_ims = sample_images(
                ddim_eta,
                ddim_steps,
                h,
                input_im,
                models,
                n_samples,
                precision,
                scale,
                w,
                x_delta,
                float(y),
                z
            )
            for image in output_ims:
                image_path = os.path.join(output_folder, f'{index}.png')
                image.save(image_path)
                index += 1

                new_image = Image.open(image_path)
                # new_image.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

                input_im = process_image(device, models, False, new_image, h, w)

            prev_x = current_x
    elif ',' in y:
        y_list = y.split(',')
        prev_y = 0
        for current_y in y_list:
            current_y = float(current_y)
            y_delta = current_y if index == 0 else current_y - prev_y
            output_ims = sample_images(
                ddim_eta,
                ddim_steps,
                h,
                input_im,
                models,
                n_samples,
                precision,
                scale,
                w,
                float(x),
                y_delta,
                z
            )
            for image in output_ims:
                image_path = os.path.join(output_folder, f'{index}.png')
                image.save(image_path)
                index += 1

                new_image = Image.open(image_path)
                new_image.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

                input_im = process_image(device, models, preprocess, new_image, h, w)

            prev_y = current_y
    else:
        output_ims = sample_images(
            ddim_eta,
            ddim_steps,
            h,
            input_im,
            models,
            n_samples,
            precision,
            scale,
            w,
            float(x),
            float(y),
            z
        )
        for image in output_ims:
            image.save(os.path.join(output_folder, f'{index}.png'))
            index += 1

    print(f'Finished creating {index} output images')


def process_image(device, models, preprocess, raw_im, h, w):
    input_im = preprocess_image(models, raw_im, preprocess)
    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])
    return input_im


def sample_images(ddim_eta, ddim_steps, h, input_im, models, n_samples, precision, scale, w, x, y, z):
    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_x = x  # NOTE: Set this way for consistency.
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims


def main():
    args = parse_args()

    device = f'cuda:0'

    # Load the input image from the path provided in command-line arguments.
    raw_im = Image.open(args.input_image)

    config = 'configs/sd-objaverse-finetune-c_concat-256.yaml'

    config = OmegaConf.load(config)

    models = init_models(config, args.ckpt_path, device)

    main_cli(
        models,
        device,
        raw_im=raw_im,
        n_samples=args.n_samples,
        x=args.polar_angle,
        y=args.azimuth_angle,
        z=args.zoom,
        preprocess=args.preprocess,
        ddim_steps=args.ddim_steps,
        output_folder=args.output_folder
    )


if __name__ == '__main__':
    main()
