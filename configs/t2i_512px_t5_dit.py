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
    cfg_indicator=0.10,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 64, 64)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,
        batch_size=1024,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.00002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    global model
    config.nnet = d(
        name='dit',
        model_args=model,
    )
    config.loss_coeffs = []
    
    config.dataset = d(
        name='JDB_demo_features',
        resolution=512,
        llm='t5',
        train_path='/data/qihao/dataset/JDB_demo_feature/',
        val_path='/data/qihao/dataset/coco_val_features/',
        cfg=False
    )

    config.sample = d(
        sample_steps=50,
        n_samples=30000,
        mini_batch_size=10,
        cfg=False,
        scale=7,
        path=''
    )

    return config
