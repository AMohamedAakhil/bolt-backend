from ip_adapter import IPAdapter
from masker import pipe, image_encoder_path, ip_ckpt, device, image_grid   
from masker import image, masked_image, mask

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=10,
                           seed=42, image=masked_image, mask_image=mask, strength=0.7, )
grid = image_grid(images, 1, 1)