import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import
from ip_adapter import IPAdapter
import intel_extension_for_pytorch as ipex

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = r"C:\Users\Aakhil\Desktop\hackathons\bolt\ComfyUI_windows_portable\ComfyUI\scripts\IP-Adapter\models\image_encoder"
ip_ckpt = r"C:\Users\Aakhil\Desktop\hackathons\bol  t\ComfyUI_windows_portable\ComfyUI\scripts\IP-Adapter\models\ip-adapter_sd15.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

sample = torch.randn(2,4,64,64)
timestep = torch.rand(1)*999
encoder_hidden_status = torch.randn(2,77,768)
input_example = (sample, timestep, encoder_hidden_status)

# OPTIMIZATIONS WITH IPEX
pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    latency = elapsed_time(pipe, prompt)
    print(latency)

image = Image.open(r"C:\Users\Aakhil\Desktop\hackathons\bolt\ComfyUI_windows_portable\ComfyUI\scripts\jacket.jpg")
image.resize((256, 256))
masked_image = Image.open(r"C:\Users\Aakhil\Desktop\hackathons\bolt\ComfyUI_windows_portable\ComfyUI\scripts\mask_image.png").resize((512, 768))
mask = Image.open(r"C:\Users\Aakhil\Desktop\hackathons\bolt\ComfyUI_windows_portable\ComfyUI\scripts\mask.png").resize((512, 768))
image_grid([masked_image.resize((256, 384)), mask.resize((256, 384))], 1, 2)