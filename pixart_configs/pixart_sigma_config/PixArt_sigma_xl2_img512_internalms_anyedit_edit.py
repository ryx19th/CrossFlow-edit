_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data/anyedit-all'
image_list_json = ['data_info.json']

edit_mode = True

data = dict(
    type='InternalDataMSSigma', root='InternData', image_list_json=image_list_json, transform='default_train', edit_mode=edit_mode,
    load_vae_feat=False, load_t5_feat=False,
)
test_data_root = 'data/anyedit-val'
test_data = dict(
    type='InternalDataMSSigma', root='InternData', image_list_json=image_list_json, transform='default_test', edit_mode=edit_mode,
    load_vae_feat=False, load_t5_feat=False,
)
image_size = 512

cond_mode = 'channel' # 'self-attn' # 'cross-attn' # ['channel', 'cross-attn', 'self-attn'] # 

# model setting
model = dict(
    type='PixArtMS_XL_2', edit_mode=edit_mode, cond_mode=cond_mode
)
mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
load_from = "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 16
train_batch_size = 48
num_epochs = 200
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
visualize = True
log_interval = 10
save_model_epochs = 1
save_model_steps = 200
work_dir = 'output/anyedit_edit_all'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.1
