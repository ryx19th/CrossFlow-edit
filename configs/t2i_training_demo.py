import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,                                               # 768 for CLIP, 4096 for T5-XXL
    num_clip_token=77,
    gradient_checking=True,
    cfg_indicator=0.1,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    stage_configs = [                                           # this is just an example
            Args(
                block_type = "TransformerBlock", 
                dim = 960,
                hidden_dim = 1920,
                num_attention_heads = 16,
                num_blocks = 29,
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 4,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 480, 
                hidden_dim = 960, 
                kernel_size = 7, 
                num_blocks = 15,
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 240, 
                hidden_dim = 480, 
                kernel_size = 7, 
                num_blocks = 15,
                max_height = 64,
                max_width = 64,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234                                          # random seed
    config.z_shape = (4, 64, 64)                                # image latent size

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth', # path of pretrained VAE CKPT from LDM
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,                                        # total training iterations
        batch_size=4,                                           # overall batch size across ALL gpus, where batch_size_per_gpu == batch_size / number_of_gpus
        mode='cond',
        log_interval=10,
        eval_interval=10,                                       # iteration interval for visual testing on the specified prompt
        save_interval=100,                                      # iteration interval for saving checkpoints and testing FID
        n_samples_eval=5,                                       # number of samples duing visual testing. This depends on your GPU memory and can be any integer between 1 and 15 (as we provide only 15 prompts).
    )

    config.optimizer = d(
        name='adamw',
        lr=0.00001,                                             # learning rate
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000                                       # warmup steps
    )

    global model
    config.nnet = d(
        name='dimr',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1/2, 1]                          # weight on loss, only needed for DiMR. Here, loss = 1/4 * loss_block1 + 1/2 * loss_block2 + 1 * loss_block3
    
    config.dataset = d(
        name='JDB_demo_features',                               # dataset name
        resolution=512,                                         # dataset resolution
        llm='clip',                                             # language model to generate language embedding
        train_path='/data/qihao/dataset/JDB_demo_feature/',     # training set path
        val_path='/data/qihao/dataset/coco_val_features/',      # val set path
        cfg=False
    )

    config.sample = d(
        sample_steps=50,                                        # sample steps duing inference/testing
        n_samples=30000,                                        # number of samples for testing (during training, we sample 10K images, which is hardcoded in the training script)
        mini_batch_size=10,                                     # batch size for testing (i.e., the number of images generated per GPU)
        cfg=False,
        scale=7,                                                # cfg scale
        path=''
    )

    return config
