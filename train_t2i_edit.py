import ml_collections
import torch
from torch import multiprocessing as mp
# from datasets import get_dataset
from datasets_img import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
# import wandb
# import numpy as np
# import time
# import random

import libs.autoencoder
from libs.t5 import T5Embedder
from libs.clip import FrozenCLIPEmbedder
from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
# from tools.fid_score import calculate_fid_given_paths
from tools.clip_score import ClipSocre

from torch.utils.tensorboard import SummaryWriter

from ipdb import set_trace as st


# PixArt imports begin

# import argparse
# import datetime
# import os
# import sys
# import time
# import types
import warnings
# from pathlib import Path

# current_file_path = Path(__file__).resolve()
# sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
# import torch
# from accelerate import Accelerator, InitProcessGroupKwargs
# from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
# from mmcv.runner import LogBuffer
# from PIL import Image
# from torch.utils.data import RandomSampler

# from pixart_diffusion import IDDPM, DPMS
# from pixart_diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from pixart_diffusion.model.builder import build_model
from pixart_diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
# from pixart_diffusion.utils.data_sampler import AspectRatioBatchSampler
# from pixart_diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
# from pixart_diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
# from pixart_diffusion.utils.lr_scheduler import build_lr_scheduler
from pixart_diffusion.utils.misc import read_config
# from pixart_diffusion.utils.misc import set_random_seed, init_random_seed, DebugUnderflowOverflow
# from pixart_diffusion.utils.optimizer import build_optimizer, auto_scale_lr

# warnings.filterwarnings("ignore")

# PixArt imports end


def enable_tf32():
    # # PyTorch 2.9+ 
    # if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    #     torch.backends.cuda.matmul.fp32_precision = "tf32"
    #     torch.backends.cudnn.conv.fp32_precision = "tf32"
    # else:
    # PyTorch <=2.8 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# enable_tf32()


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    # ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator()  # kwargs_handlers=[ddp_kwargs]
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
        #            name=config.hparams, job_type='train', mode='offline')
        writer = SummaryWriter(log_dir=os.path.abspath(config.workdir))
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    # assert os.path.exists(dataset.fid_stat)

    gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    num_workers = 8

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                    num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=False, drop_last=False,
                                     num_workers=num_workers, pin_memory=True, persistent_workers=True)

    nnet = None
    # st()
    if config.get('base_net', None) == 'pixart':
        print('Use PixArt base net and loading its config...')
        pixart_config = read_config('pixart_configs/pixart_sigma_config/PixArt_sigma_xl2_img256_internal_anyedit_edit.py')
        image_size = pixart_config.image_size  # @param [256, 512]
        latent_size = int(image_size) // 8
        # pred_sigma = getattr(pixart_config, 'pred_sigma', True)
        # learn_sigma = getattr(pixart_config, 'learn_sigma', True) and pred_sigma
        max_length = pixart_config.model_max_length
        vae = AutoencoderKL.from_pretrained(pixart_config.vae_pretrained, torch_dtype=torch.float16).to(accelerator.device)
        pixart_config.scale_factor = vae.config.scaling_factor
        pipeline_load_from = 'pixart_pretrained_models/pixart_sigma_sdxlvae_T5_diffusers'
        tokenizer = T5Tokenizer.from_pretrained(pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

        model_kwargs = {"pe_interpolation": pixart_config.pe_interpolation, "config": pixart_config,
                        "model_max_length": max_length, "qk_norm": pixart_config.qk_norm,
                        "kv_compress_config": None, "micro_condition": pixart_config.micro_condition}

        # build models
        model = build_model(pixart_config.model,
                            pixart_config.grad_checkpointing,
                            pixart_config.get('fp32_attention', False),
                            input_size=latent_size,
                            learn_sigma=False,
                            pred_sigma=False,
                            **model_kwargs)

        if pixart_config.load_from is not None:
            missing, unexpected = load_checkpoint(
                pixart_config.load_from, model, load_ema=pixart_config.get('load_ema', False), max_length=max_length)
            logging.warning(f'Missing keys: {missing}')
            logging.warning(f'Unexpected keys: {unexpected}')

        nnet = model
        # autoencoder = vae
    # st()

    train_state = utils.initialize_train_state(config, device, nnet=nnet)

    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    # st()
    lr_scheduler = train_state.lr_scheduler

    # st()
    if config.get('base_net', None) == 'pixart':
        pass
        # st()
        # config.resume_from = dict(
        #     checkpoint=args.resume_from,
        #     load_ema=False,
        #     resume_optimizer=True,
        #     resume_lr_scheduler=True)
        # resume_path = config.resume_from['checkpoint']
        # path = os.path.basename(resume_path)
        # start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        # start_step = int(path.replace('.pth', '').split("_")[3])
        # _, missing, unexpected = load_checkpoint(**config.resume_from,
        #                                         model=model,
        #                                         optimizer=optimizer,
        #                                         lr_scheduler=lr_scheduler,
        #                                         max_length=max_length,
        #                                         )

        # logger.warning(f'Missing keys: {missing}')
        # logger.warning(f'Unexpected keys: {unexpected}')
    elif config.get('base_net', None) is None:
        if config.get('load_from', '') != '':
            train_state.load_pretrained(config.load_from)
        train_state.resume(config.ckpt_root)

    if config.get('base_net', None) is None:
        autoencoder = libs.autoencoder.get_model(**config.autoencoder)
        autoencoder.to(device)

    if config.get('base_net', None) is None:
        t5 = clip = None
        if config.nnet.model_args.do_class_cond:
            llm = None
        else:
            if config.nnet.model_args.clip_dim == 4096: # this way # 
                llm = "t5"
                t5 = T5Embedder(device=device)
            elif config.nnet.model_args.clip_dim == 768:
                llm = "clip"
                clip = FrozenCLIPEmbedder()
                clip.eval()
                clip.to(device)
            else:
                raise NotImplementedError
    else:
        llm = t5 = clip = None

    ss_empty_context = None

    ClipSocre_model = ClipSocre(device=device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def encode_moments(_batch):
        return autoencoder(_batch, fn='encode_moments').squeeze(0)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    @torch.cuda.amp.autocast()
    def pixart_encode(_batch):
        posterior = vae.encode(_batch).latent_dist
        if pixart_config.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z * pixart_config.scale_factor

    @torch.cuda.amp.autocast()
    def pixart_decode(_batch):
        # _batch = _batch.to(torch.float16)
        samples = vae.decode(_batch.detach() / pixart_config.scale_factor).sample
        return samples

    # @torch.cuda.amp.autocast()
    def encode_text_t5(_batch):
        _latent_t5, latent_and_others_t5 = t5.get_text_embeddings(_batch)
        token_embedding_t5 = latent_and_others_t5['token_embedding'].to(torch.float32) * 10.0
        token_mask_t5 = latent_and_others_t5['token_mask']
        token_t5 = latent_and_others_t5['tokens']
        # st()
        return token_embedding_t5, token_mask_t5, token_t5
    
    # @torch.cuda.amp.autocast()
    def encode_text_clip(_batch):
        _latent_clip, latent_and_others_clip = clip.encode(_batch)
        token_embedding_clip = latent_and_others_clip['token_embedding']
        token_mask_clip = latent_and_others_clip['token_mask']
        token_clip = latent_and_others_clip['tokens']
        return token_embedding_clip, token_mask_clip, token_clip

    def encode_all_prompts(_batch, llm_fn, do_regular_cfg):
        if isinstance(_batch[0], list):
            assert len(_batch) == 2
            _batch_con, _batch_mask, _batch_token = llm_fn(_batch[0])
            _batch_con_src, _batch_mask_src, _batch_token_src = llm_fn(_batch[1])
            _batch_con = torch.cat([_batch_con, _batch_con_src], dim=1)
            _batch_mask = torch.cat([_batch_mask, _batch_mask_src], dim=1)
            _batch_token = torch.cat([_batch_token, _batch_token_src], dim=1)
        else:
            _batch_con, _batch_mask, _batch_token = llm_fn(_batch)
        if do_regular_cfg:
            _null_con, _null_mask, _null_token = llm_fn([''])
            if isinstance(_batch[0], list):
                _null_con = torch.cat([_null_con, _null_con], dim=1)
                _null_mask = torch.cat([_null_mask, _null_mask], dim=1)
                _null_token = torch.cat([_null_token, _null_token], dim=1)
            _null_context = {'cond': _null_con, 'con_mask': _null_mask, 'text_token': _null_token}
        else:
            _null_context = None
        # st()
        return _batch_con, _batch_mask, _batch_token, _null_context
    
    def pixart_encode_text_helper(_batch):
        txt_tokens = tokenizer(
            _batch, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to(accelerator.device)
        y = text_encoder(
            txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
        y_mask = txt_tokens.attention_mask[:, None, None]
        return txt_tokens, y, y_mask
    
    def pixart_encode_text(_batch, do_regular_cfg):
        if isinstance(_batch[0], list):
            assert len(_batch) == 2
            txt_tokens, y, y_mask = pixart_encode_text_helper(_batch[0])
            txt_tokens_src, y_src, y_mask_src = pixart_encode_text_helper(_batch[1])
            for key in txt_tokens.keys():
                txt_tokens[key] = torch.cat([txt_tokens[key], txt_tokens_src[key]], dim=-1)
            y = torch.cat([y, y_src], dim=-2)
            y_mask = torch.cat([y_mask, y_mask_src], dim=-1)
        else:
            txt_tokens, y, y_mask = pixart_encode_text_helper(_batch)
        if do_regular_cfg:
            null_txt_tokens, null_y, null_y_mask = pixart_encode_text_helper([''])
            if isinstance(_batch[0], list):
                for key in null_txt_tokens.keys():
                    null_txt_tokens[key] = torch.cat([null_txt_tokens[key], null_txt_tokens[key]], dim=-1)
                null_y = torch.cat([null_y, null_y], dim=-2)
                null_y_mask = torch.cat([null_y_mask, null_y_mask], dim=-1)
            _null_context = {'cond': null_y, 'con_mask': null_y_mask, 'text_token': null_txt_tokens}
        else:
            _null_context = None
        return y, y_mask, txt_tokens, _null_context

    def get_data_generator(dataloader):
        while True:
            for data in tqdm(dataloader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    train_data_generator = get_data_generator(train_dataset_loader)
    # test_data_generator = get_data_generator(test_dataset_loader)

    # def get_context_generator(autoencoder):
    #     while True:
    #         for data in test_dataset_loader:
    #             if len(data) == 5:
    #                 _img, _context, _token_mask, _token, _caption = data
    #             else:
    #                 _img, _context = data
    #                 _token_mask = None
    #                 _token = None
    #                 _caption = None
                
    #             if len(_img.shape)==5:
    #                 _testbatch_img_blurred = autoencoder.sample(_img[:,1,:]) 
    #                 yield _context, _token_mask, _token, _caption, _testbatch_img_blurred
    #             else:
    #                 assert len(_img.shape)==4
    #                 yield _context, _token_mask, _token, _caption, None

    # context_generator = get_context_generator(autoencoder)

    _flow_mathcing_model = FlowMatching()

    def train_step(_batch, _ss_empty_context, llm, t5, clip):
        _metrics = dict()
        optimizer.zero_grad()

        # st()

        edit_mode = config.get('edit_mode', False)
        do_regular_cfg = (nnet.module.do_regular_cfg if hasattr(nnet.module, "do_regular_cfg") else False) if config.get('base_net', None) is None else True

        assert len(_batch) == 3 if edit_mode else 2
        assert not config.dataset.cfg
        # _batch_img = _batch[0]
        # _batch_con = _batch[1]
        # _batch_mask = _batch[2]
        # _batch_token = _batch[3]
        # _batch_caption = _batch[4]
        # _batch_img_ori = _batch[5]
        _batch_img_ori = _batch[0]
        _batch_caption = _batch[1]
        if edit_mode:
            _batch_img_src_ori = _batch[2]
        # logging.info(_batch_caption)

        # st()
        if config.get('base_net', None) == 'pixart':
            with torch.no_grad():
                _z = pixart_encode(_batch_img_ori)
                if edit_mode:
                    _z_cond = pixart_encode(_batch_img_src_ori)

                _batch_con, _batch_mask, _batch_token, _null_context = pixart_encode_text(_batch_caption, do_regular_cfg)
            # st()

        elif config.get('base_net', None) is None:
            # _batch_img = encode_moments(_batch_img_ori)
            _z = encode(_batch_img_ori)
            if edit_mode:
                _z_cond = encode(_batch_img_src_ori)

                if config.dataset.naive_mode == 'hole_latent':
                    _z_hole = _z_cond.clone()
                    start_idx = _z_hole.shape[-1] // 4
                    end_idx = _z_hole.shape[-1] * 3 // 4
                    _z_hole[:, :, start_idx:end_idx, start_idx:end_idx] = -1 # not sure how to make full black hole on latent 
                    _z = _z_hole

            if llm is None:
                _batch_con = _batch_caption
                _batch_mask, _batch_token, _null_context = None, None, None
            else:
                if llm == 't5':
                    llm_fn = encode_text_t5
                elif llm == 'clip':
                    llm_fn = encode_text_clip
                _batch_con, _batch_mask, _batch_token, _null_context = encode_all_prompts(_batch_caption, llm_fn, do_regular_cfg)

        use_textVE = not (edit_mode and nnet.module.cond_mode == 'cross-attn' and nnet.module.direct_map and nnet.module.use_cross_attn) if config.get('base_net', None) is None else False

        loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_batch_con, con_mask=_batch_mask, batch_img_clip=_batch_img_ori, \
            nnet_style=config.nnet.name, text_token=_batch_token, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step, cond_image=_z_cond if edit_mode else None, use_textVE=use_textVE, _null_context=_null_context, use_PixArt=config.get('base_net', None) == 'pixart', do_class_cond=config.nnet.model_args.do_class_cond, )

        ## 
        flag = (~torch.isfinite(loss)).any().to(dtype=torch.int, device=loss.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
        world_has_bad = flag.item() > 0
        if world_has_bad:
            logging.info(f"[skip] non-finite loss at step {train_state.step}")
            _metrics = {}
            _metrics['loss'] = torch.nanmean(accelerator.gather(loss.detach()))
            for k, v in loss_dict.items():
                _metrics[k] = torch.nanmean(accelerator.gather(v.detach()))
            return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
        ## 

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        for key in loss_dict.keys():
            _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
        accelerator.backward(loss.mean())

        max_norm = config.get('clip_grad_norm', None)
        # st()
        if max_norm is None or max_norm <= 0:
            max_norm = 1e9 # just for display 
        _metrics['max_norm'] = max_norm
        grad_norm = accelerator.clip_grad_norm_(nnet.parameters(), max_norm)
        _metrics['grad_norm'] = grad_norm.item()
        # torch.nn.utils.get_total_norm 
        grad_norm_clipped = accelerator.clip_grad_norm_(nnet.parameters(), max_norm=1e9) # just for display 
        _metrics['grad_norm_clipped'] = grad_norm_clipped.item()

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token_mask=None, return_clipScore=False, ClipSocre_model=None, cond_image=None, use_textVE=True, _null_context=None, use_PixArt=False, do_class_cond=False):
        with torch.no_grad():

            if use_textVE:

                _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)
                    
                _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
                _z_init = _z_x0.reshape(_z_gaussian.shape)

            else:
                # st()
                _z_init = context
            # /\ 
            if use_PixArt or (nnet_ema.module.edit_mode and nnet_ema.module.direct_map) : 
                # st()
                _z_init, cond_image = cond_image, _z_init

            # assert config.sample.scale >= 1
            _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator") if not use_PixArt else False
            do_regular_cfg = (config.nnet.model_args.do_regular_cfg if hasattr(config.nnet.model_args, "do_regular_cfg") else False) if not use_PixArt else True

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg ) 
            # st() 
            _z, _ = ode_solver.sample(x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator, cond_image=cond_image, cond_mask=token_mask, do_regular_cfg=do_regular_cfg, _null_context=_null_context, use_PixArt=use_PixArt, do_class_cond=do_class_cond,)

            if use_PixArt:
                image_unprocessed = pixart_decode(_z)
            else:
                image_unprocessed = decode(_z)

            if return_clipScore:
                clip_score = ClipSocre_model.calculate_clip_score(caption, image_unprocessed)
                return image_unprocessed, clip_score
            else:
                return image_unprocessed

    # def eval_step(n_samples, sample_steps):
    #     logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
    #                  f'mini_batch_size={config.sample.mini_batch_size}')
        
    #     def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
    #         _context, _token_mask, _token, _caption, _testbatch_img_blurred = next(context_generator)
    #         assert _context.size(0) == _n_samples
    #         assert not return_caption # during training we should not use this 
    #         if return_caption:
    #             return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask), _caption
    #         elif return_clipScore:
    #             return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, caption=_caption)
    #         else:
    #             return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask)

    #     with tempfile.TemporaryDirectory() as temp_path:
    #         path = config.sample.path or temp_path
    #         if accelerator.is_main_process:
    #             os.makedirs(path, exist_ok=True)
    #         clip_score_list = utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, return_clipScore=True, ClipSocre_model=ClipSocre_model, config=config)
    #         _fid = 0
    #         if accelerator.is_main_process:
    #             _fid = calculate_fid_given_paths((dataset.fid_stat, path))
    #             _clip_score_list = torch.cat(clip_score_list)
    #             logging.info(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}')
    #             with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
    #                 print(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}', file=f)
    #             wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
    #         _fid = torch.tensor(_fid, device=device)
    #         _fid = accelerator.reduce(_fid, reduction='sum')

    #     return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    # step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x, next(train_data_generator))
        metrics = train_step(batch, ss_empty_context, llm, t5, clip)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            # wandb.log(metrics, step=train_state.step)
            for k, v in metrics.items():
                writer.add_scalar(k, v, train_state.step)

        ############# save rigid image
        if train_state.step % config.train.eval_interval == 0 or train_state.step == 1 :
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            edit_mode = config.get('edit_mode', False)
            do_regular_cfg = (nnet_ema.module.do_regular_cfg if hasattr(nnet_ema.module, "do_regular_cfg") else False) if config.get('base_net', None) is None else True
            # st()
            # batch = tree_map(lambda x: x, next(test_data_generator))
            # for batch in test_dataset_loader: break 
            # batch = test_dataset[:config.train.n_samples_eval]
            batch = [[], [], []] if edit_mode else [[], []]
            invertal = len(test_dataset) // (config.train.n_samples_eval - 1) 
            for i in range(config.train.n_samples_eval):
                idx = i * invertal 
                batch_raw = test_dataset[idx]
                batch[0].append(torch.tensor(batch_raw[0], device=device))
                batch[1].append(batch_raw[1])
                if edit_mode:
                    batch[2].append(torch.tensor(batch_raw[2], device=device))
            # st()
            batch[0] = torch.stack(batch[0], dim=0)
            _batch_img_ori = batch[0]
            _batch_caption = batch[1]
            print(_batch_caption)
            if isinstance(_batch_caption[0], list):
                _batch_caption_ = [[], []]
                for _b_c in _batch_caption:
                    _batch_caption_[0].append(_b_c[0])
                    _batch_caption_[1].append(_b_c[1])
                _batch_caption = _batch_caption_
            elif isinstance(_batch_caption[0], torch.Tensor):
                _batch_caption = torch.stack(_batch_caption)
            if edit_mode:
                batch[2] = torch.stack(batch[2], dim=0)
                _batch_img_src_ori = batch[2]

            # st()
            if config.get('base_net', None) == 'pixart':
                with torch.no_grad():
                    if edit_mode:
                        _z_cond = pixart_encode(_batch_img_src_ori)

                    _batch_con, _batch_mask, _batch_token, _null_context = pixart_encode_text(_batch_caption, do_regular_cfg)
                # st()

            elif config.get('base_net', None) is None:
                # _z = encode(_batch_img_ori)
                if edit_mode:
                    _z_cond = encode(_batch_img_src_ori)

                    if config.dataset.naive_mode == 'hole_latent':
                        _z_hole = _z_cond.clone()
                        start_idx = _z_hole.shape[-1] // 4
                        end_idx = _z_hole.shape[-1] * 3 // 4
                        _z_hole[:, :, start_idx:end_idx, start_idx:end_idx] = -1 # not sure how to make full black hole on latent 
                        _z = _z_hole

                if llm is None:
                    _batch_con = _batch_caption
                    _batch_mask, _batch_token, _null_context = None, None, None
                else:
                    if llm == 't5':
                        llm_fn = encode_text_t5
                    elif llm == 'clip':
                        llm_fn = encode_text_clip
                    _batch_con, _batch_mask, _batch_token, _null_context = encode_all_prompts(_batch_caption, llm_fn, do_regular_cfg)

            contexts = _batch_con
            token_mask = _batch_mask
            # if hasattr(dataset, "token_embedding"):
            #     contexts = torch.tensor(dataset.token_embedding, device=device)[ : config.train.n_samples_eval]
            #     token_mask = torch.tensor(dataset.token_mask, device=device)[ : config.train.n_samples_eval]
            # elif hasattr(dataset, "contexts"):
            #     contexts = torch.tensor(dataset.contexts, device=device)[ : config.train.n_samples_eval]
            #     token_mask = None
            # else:
            #     raise NotImplementedError
            use_textVE = not (edit_mode and nnet_ema.module.cond_mode == 'cross-attn' and nnet_ema.module.direct_map and nnet_ema.module.use_cross_attn) if config.get('base_net', None) is None else False
            # st()
            samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, context=contexts, token_mask=token_mask, cond_image=_z_cond if edit_mode else None, use_textVE=use_textVE, _null_context=_null_context, use_PixArt=config.get('base_net', None) == 'pixart', do_class_cond=config.nnet.model_args.do_class_cond, )
            # st()
            samples = dataset.unpreprocess(samples)
            gt_samples = dataset.unpreprocess(_batch_img_ori)
            if edit_mode:
                src_samples = dataset.unpreprocess(_batch_img_src_ori)
                all_samples = torch.stack([src_samples, samples, gt_samples], dim=1 )
            else:
                all_samples = torch.stack([samples, gt_samples], dim=1 )
            all_samples = einops.rearrange(all_samples, 'b n c h w -> (b n) c h w')
            all_samples = make_grid(all_samples, nrow=3 if edit_mode else 2)
            if accelerator.is_main_process:
                save_image(all_samples, os.path.join(config.sample_dir, f'{train_state.step}.jpg'))
                # wandb.log({'samples': wandb.Image(all_samples)}, step=train_state.step)
                writer.add_image("samples", all_samples, train_state.step)
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        ############ save checkpoint and evaluate results
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')

            if accelerator.local_process_index == 0:
                if config.get('base_net', None) == 'pixart':
                    os.umask(0o000)
                    save_checkpoint(config.ckpt_root,
                                    epoch=train_state.step // (len(train_dataset) // config.train.batch_size),
                                    step=train_state.step,
                                    model=accelerator.unwrap_model(nnet),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
                elif config.get('base_net', None) is None:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

            # fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            # step_fid.append((train_state.step, fid))

            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    # logging.info(f'step_fid: {step_fid}')
    # step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    # logging.info(f'step_best: {step_best}')
    # train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    # eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    if FLAGS.workdir is not None:
        config.workdir = FLAGS.workdir
    elif config.get('workdir', None) is not None:
        config.workdir = os.path.join('workdir', config.workdir, config.hparams)
    else:
        config.workdir = os.path.join('workdir', config.config_name, config.hparams)
    print('Workdir: ', config.workdir)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
