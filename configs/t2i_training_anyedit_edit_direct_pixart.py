import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    latent_size = 64,
    learn_sigma = False, # different from DiT, we direct predict noise here
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=4096,
    num_clip_token=77,
    gradient_checking=True, # for larger model
    # cfg_indicator=0.10,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    edit_mode=True,
    cond_mode='cross-attn', # 'channel', # 'self-attn', #              # ['channel', 'cross-attn', 'self-attn'] 
    direct_map=True,
    use_cross_attn=True, # False, # 
    do_regular_cfg=True, # False, # 
    prompt_dropout_prob=0.0, # 0.1, # 
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234                                          # random seed
    config.z_shape = (4, 32, 32) # (4, 64, 64)                                # image latent size

    # config.autoencoder = d(
    #     pretrained_path='assets/stable-diffusion/autoencoder_kl.pth', # path of pretrained VAE CKPT from LDM
    #     scale_factor=0.23010
    # )

    config.train = d(
        n_steps=100000,                                        # total training iterations
        batch_size=2048, # 1024, # 16, #                                          # overall batch size across ALL gpus, where batch_size_per_gpu == batch_size / number_of_gpus
        mode='cond',
        log_interval=1, # 10, # 
        eval_interval=10, # 100, #                                      # iteration interval for visual testing on the specified prompt
        save_interval=100, # 5000, #                                     # iteration interval for saving checkpoints and testing FID
        n_samples_eval=8,                                       # number of samples duing visual testing. This depends on your GPU memory and can be any integer between 1 and 15 (as we provide only 15 prompts).
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0001, # 0.00001, # 0.00002, #                                            # learning rate
        weight_decay=0.0, # 0.03, # 
        betas=(0.9, 0.9), # (0.9, 0.999, 0.9999), # 
        # eps=(1e-30, 1e-16), # 
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=100, # 1000, # 5000, #                                      # warmup steps
    )

    global model
    config.nnet = d(
        name='dit', # 'dimr', # 
        model_args=model,
    )
    config.loss_coeffs = [] # [1/4, 1/2, 1] #                         # weight on loss, only needed for DiMR. Here, loss = 1/4 * loss_block1 + 1/2 * loss_block2 + 1 * loss_block3
    
    config.dataset = d(
        name='ImageDataset',                               # dataset name
        resolution=256, # 512, #                                          # dataset resolution
        llm='t5', # 'clip', #                                            # language model to generate language embedding
        train_path='anyedit', # 'anyedit_1k', # 'anyedit_all', #     # training set path
        val_path='anyedit', # 'anyedit_1k', # 'anyedit_val', #                     # val set path
        cfg=False,
        edit_mode=True,
        prompt_mode='dual', # 'output', # 'instruction', #               # ['output', 'dual', 'instruction']
    )

    config.sample = d(
        sample_steps=50,                                        # sample steps duing inference/testing
        n_samples=30000,                                        # number of samples for testing (during training, we sample 10K images, which is hardcoded in the training script)
        mini_batch_size=10,                                     # batch size for testing (i.e., the number of images generated per GPU)
        cfg=False,
        scale=1.0, # 4.5, #                                               # cfg scale
        path=''
    )

    # config.load_from = 'pretrained_models/t2i_512px_t5_dit.pth' # 'pretrained_models/t2i_512px_clip_dimr.pth' # 

    config.edit_mode = True

    config.workdir = 't2i_training_anyedit_edit_direct_pixart_dualp' # 't2i_training_anyedit_edit_direct_pixart' # 't2i_training_anyedit_edit_direct_pixart_overfit_dualp' # 't2i_training_anyedit_edit_direct_pixart_overfit' # 't2i_training_anyedit_edit_direct_pixart_overfit_instp' # 't2i_training_anyedit_edit_direct_pixart_instp' # 

    config.base_net = 'pixart'

    return config




# editclip