import argparse
import torch
import os
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

# OPTIMIZATION: Import optimized components from demo.py
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

# OPTIMIZATION: Aggressive memory cleanup at start
torch.cuda.empty_cache()
import gc
gc.collect()

free_vram_gb = get_cuda_free_memory_gb(gpu)
print(f'Free VRAM {free_vram_gb} GB')

# OPTIMIZATION: Enhanced memory mode detection (from demo.py)
if free_vram_gb < 6.0:
    print("⚠️ Using minimal memory mode due to very limited VRAM")
    low_memory = True
    minimal_memory = True
elif free_vram_gb < 20.0:
    print("⚠️ Using optimized low memory mode for 15GB VRAM (encoder+VAE on GPU, transformer dynamic)")
    low_memory = True
    minimal_memory = False
else:
    print("✅ Using high memory mode (all models on GPU)")
    low_memory = False
    minimal_memory = False

print(f"✅ Memory mode: {'minimal' if minimal_memory else ('optimized low' if low_memory else 'high')} memory mode")

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# OPTIMIZATION: Load models using optimized wrappers (like demo.py)
print("Loading text encoder...")
print("  - Initializing UMT5 model...")
torch.cuda.empty_cache()
text_encoder = WanTextEncoder()
print("✅ Text encoder loaded successfully")

print("Loading transformer model...")
print("  - Initializing WanDiffusionWrapper...")
transformer = WanDiffusionWrapper(is_causal=True)
print("  - Loading checkpoint weights...")

# OPTIMIZATION: Optimized checkpoint loading (from demo.py)
if args.checkpoint_path:
    if low_memory:
        print("  - Using CPU loading due to low memory...")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    else:
        print("  - Loading directly to GPU...")
        state_dict = torch.load(args.checkpoint_path, map_location=gpu)

    print("  - Applying weights to model...")
    # Check what keys are available in the checkpoint
    print(f"📋 Checkpoint keys: {list(state_dict.keys())}")
    
    # Use generator_ema by default since that's what's available
    if 'generator_ema' in state_dict:
        print("✅ Loading generator_ema from checkpoint")
        transformer.load_state_dict(state_dict['generator_ema'])
    elif 'generator' in state_dict:
        print("✅ Loading generator from checkpoint")
        transformer.load_state_dict(state_dict['generator'])
    else:
        raise KeyError(f"No generator key found in checkpoint. Available keys: {list(state_dict.keys())}")

    # Clean up state dict to free memory
    del state_dict
    gc.collect()
print("✅ Transformer model loaded successfully")

# OPTIMIZATION: Initialize VAE decoder using optimized wrapper
print("Loading VAE decoder...")
vae_decoder = VAEDecoderWrapper()

# OPTIMIZATION: Optimized VAE loading (from demo.py)
if low_memory:
    print("  - Using CPU loading for VAE due to low memory...")
    vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
else:
    print("  - Loading VAE directly to GPU...")
    vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location=gpu)

decoder_state_dict = {}
for key, value in vae_state_dict.items():
    if 'decoder.' in key or 'conv2' in key:
        decoder_state_dict[key] = value
vae_decoder.load_state_dict(decoder_state_dict)

# Clean up to free memory
del vae_state_dict, decoder_state_dict
gc.collect()

vae_decoder.eval()
vae_decoder.to(dtype=torch.bfloat16)
vae_decoder.requires_grad_(False)

# Only move to GPU if not already there (for CPU-loaded models)
if low_memory:
    print("  - Moving VAE decoder to GPU (loaded from CPU)...")
    vae_decoder.to(gpu)
else:
    print("  - VAE decoder already on GPU (loaded directly)")

print("✅ VAE decoder loaded successfully")

print("Setting up model configurations...")
text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.bfloat16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

print("Initializing pipeline...")
# OPTIMIZATION: Use GPU for pipeline device
pipeline_device = gpu
# Initialize pipeline using optimized components
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(
        config,
        device=pipeline_device,
        generator=transformer,
        text_encoder=text_encoder,
        vae=vae_decoder
    )
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(
        config,
        device=pipeline_device,
        generator=transformer,
        text_encoder=text_encoder,
        vae=vae_decoder
    )

print("Moving models to GPU...")
# OPTIMIZATION: Clear memory before moving models
torch.cuda.empty_cache()

if minimal_memory:
    print("  - MINIMAL MEMORY MODE: Only VAE on GPU, others move dynamically...")
    vae_decoder.to(gpu)
    print("  - VAE decoder moved to GPU (essential for output)")
    print("  - Text encoder staying on CPU (moved temporarily during generation)")
    print("  - Transformer staying on CPU (moved temporarily during generation)")
elif low_memory:
    print("  - OPTIMIZED LOW MEMORY MODE: Text encoder + VAE on GPU, transformer dynamic...")
    text_encoder.to(gpu)
    print("  - Text encoder moved to GPU (~6GB)")
    print("  - Transformer staying on CPU (moved dynamically during generation)")
    vae_decoder.to(gpu)
    print("  - VAE decoder moved to GPU (~1-2GB)")
    estimated_usage = 6 + 1.5  # Text encoder + VAE only
    print(f"  - Estimated GPU usage: ~{estimated_usage}GB base models")
    print(f"  - Peak usage during generation: ~{estimated_usage + 6}GB (transformer temporarily loaded)")
else:
    print("  - HIGH MEMORY MODE: Moving all models to GPU...")
    text_encoder.to(gpu)
    transformer.to(gpu)
    vae_decoder.to(gpu)
    print("  - All models moved to GPU")

# OPTIMIZATION: Apply DynamicSwapInstaller for memory optimization
if low_memory or minimal_memory:
    print("  - Installing DynamicSwapInstaller for text encoder...")
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)

# Final memory cleanup
torch.cuda.empty_cache()
print("✅ All models loaded and ready!")


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# OPTIMIZATION: Pre-initialize VAE cache
vae_cache = ZERO_VAE_CACHE
for i in range(len(vae_cache)):
    vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.bfloat16)
print(f"✅ VAE cache initialized: {len(vae_cache)} items")


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    
    print(f"\n🎬 Processing prompt {i+1}/{num_prompts} (index {idx})")

    # OPTIMIZATION: Move transformer to GPU before generation if in memory-optimized mode
    if minimal_memory or low_memory:
        print(f"  - Moving transformer to GPU for generation...")
        transformer.to(gpu)
        torch.cuda.empty_cache()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples
        print(f"📝 Text prompt: {prompt[:100]}...")
        print(f"🖼️ Processing image-to-video generation...")

        # Process the image
        print("  - Loading image to tensor...")
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        print("  - Encoding image as initial latent...")
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
        print(f"  ✅ Image encoded: {initial_latent.shape}")

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
            print(f"📝 Extended prompt: {extended_prompt[:100]}...")
        else:
            prompts = [prompt] * args.num_samples
            print(f"📝 Text prompt: {prompt[:100]}...")
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    print(f"🎲 Generated noise tensor: {sampled_noise.shape}")
    
    # Clear GPU memory before generation
    print("🧹 Clearing GPU memory before generation...")
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    free_vram = get_cuda_free_memory_gb(gpu)
    print(f"✅ GPU memory cleared. Free VRAM: {free_vram:.2f} GB")

    # Generate frames
    print("🚀 Starting video generation...")
    generation_start_time = time.time()
    
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    
    generation_time = time.time() - generation_start_time
    print(f"⚡ Generation completed in {generation_time:.2f}s")
    
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    print("🎨 Processing final video...")
    video = 255.0 * torch.cat(all_video, dim=1)
    print(f"✅ Video processed: {video.shape}")

    # Clear VAE cache
    print("🧹 Clearing VAE cache...")
    pipeline.vae.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            print(f"💾 Saving video to: {output_path}")
            write_video(output_path, video[seed_idx], fps=16)
            print(f"✅ Video saved successfully!")
    
    # OPTIMIZATION: Move transformer back to CPU if in memory-optimized mode
    if minimal_memory or low_memory:
        print(f"  - Moving transformer back to CPU after generation...")
        transformer.to('cpu')
        torch.cuda.empty_cache()
        free_vram = get_cuda_free_memory_gb(gpu)
        print(f"  - GPU memory freed. Free VRAM: {free_vram:.2f} GB")
    
    # Clear GPU memory after each generation
    print("🧹 Clearing GPU memory after generation...")
    torch.cuda.empty_cache()
    print(f"✅ GPU memory cleared. Free VRAM: {get_cuda_free_memory_gb(gpu):.2f} GB")
