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
import wandb
import numpy as np
import time
import random

import libs.autoencoder
from libs.t5 import T5Embedder
from libs.clip import FrozenCLIPEmbedder
from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
from tools.fid_score import calculate_fid_given_paths
from tools.clip_score import ClipSocre

from ipdb import set_trace as st


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
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
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
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

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler

    # st()
    if config.get('load_from', '') != '':
        train_state.load_pretrained(config.load_from)
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    t5 = clip = None
    if config.nnet.model_args.clip_dim == 4096:
        llm = "t5"
        t5 = T5Embedder(device=device)
    elif config.nnet.model_args.clip_dim == 768:
        llm = "clip"
        clip = FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
    else:
        raise NotImplementedError

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

    def get_data_generator(dataloader):
        while True:
            for data in tqdm(dataloader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    train_data_generator = get_data_generator(train_dataset_loader)
    # test_data_generator = get_data_generator(test_dataset_loader)

    def get_context_generator(autoencoder):
        while True:
            for data in test_dataset_loader:
                if len(data) == 5:
                    _img, _context, _token_mask, _token, _caption = data
                else:
                    _img, _context = data
                    _token_mask = None
                    _token = None
                    _caption = None
                
                if len(_img.shape)==5:
                    _testbatch_img_blurred = autoencoder.sample(_img[:,1,:]) 
                    yield _context, _token_mask, _token, _caption, _testbatch_img_blurred
                else:
                    assert len(_img.shape)==4
                    yield _context, _token_mask, _token, _caption, None

    context_generator = get_context_generator(autoencoder)

    _flow_mathcing_model = FlowMatching()

    def train_step(_batch, _ss_empty_context, llm, t5, clip):
        _metrics = dict()
        optimizer.zero_grad()

        # st()

        edit_mode = config.get('edit_mode', False)

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

        # _batch_img = encode_moments(_batch_img_ori)
        _z = encode(_batch_img_ori)
        if edit_mode:
            _z_cond = encode(_batch_img_src_ori)

        if llm == 't5':
            _batch_con, _batch_mask, _batch_token = encode_text_t5(_batch_caption)
        elif llm == 'clip':
            _batch_con, _batch_mask, _batch_token = encode_text_clip(_batch_caption)
        # st()

        loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_batch_con, con_mask=_batch_mask, batch_img_clip=_batch_img_ori, \
            nnet_style=config.nnet.name, text_token=_batch_token, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step, cond_image=_z_cond if edit_mode else None)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        for key in loss_dict.keys():
            _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token_mask=None, return_clipScore=False, ClipSocre_model=None, cond_image=None):
        with torch.no_grad():
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)
                
            _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
            _z_init = _z_x0.reshape(_z_gaussian.shape)
            
            assert config.sample.scale > 1
            _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg ) 
            # st() 
            _z, _ = ode_solver.sample(x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator, cond_image=cond_image )

            image_unprocessed = decode(_z)

            if return_clipScore:
                clip_score = ClipSocre_model.calculate_clip_score(caption, image_unprocessed)
                return image_unprocessed, clip_score
            else:
                return image_unprocessed

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')
        
        def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
            _context, _token_mask, _token, _caption, _testbatch_img_blurred = next(context_generator)
            assert _context.size(0) == _n_samples
            assert not return_caption # during training we should not use this 
            if return_caption:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask), _caption
            elif return_clipScore:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, caption=_caption)
            else:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            clip_score_list = utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, return_clipScore=True, ClipSocre_model=ClipSocre_model, config=config)
            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                _clip_score_list = torch.cat(clip_score_list)
                logging.info(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

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
            wandb.log(metrics, step=train_state.step)

        ############# save rigid image
        if train_state.step % config.train.eval_interval == 0 or train_state.step == 1 :
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            edit_mode = config.get('edit_mode', False)
            # st()
            # batch = tree_map(lambda x: x, next(test_data_generator))
            # for batch in test_dataset_loader: break 
            # batch = test_dataset[:config.train.n_samples_eval]
            batch = [[], [], []] if edit_mode else [[], []]
            for i in range(config.train.n_samples_eval):
                batch[0].append(torch.tensor(test_dataset[i][0], device=device))
                batch[1].append(test_dataset[i][1])
                if edit_mode:
                    batch[2].append(torch.tensor(test_dataset[i][2], device=device))
            # st()
            batch[0] = torch.stack(batch[0], dim=0)
            _batch_img_ori = batch[0]
            # _z = encode(_batch_img_ori)
            if edit_mode:
                batch[2] = torch.stack(batch[2], dim=0)
                _batch_img_src_ori = batch[2]
                _z_cond = encode(_batch_img_src_ori)
            _batch_caption = batch[1]
            print(_batch_caption)
            if llm == 't5':
                _batch_con, _batch_mask, _batch_token = encode_text_t5(_batch_caption)
            elif llm == 'clip':
                _batch_con, _batch_mask, _batch_token = encode_text_clip(_batch_caption)
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
            samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, context=contexts, token_mask=token_mask, cond_image=_z_cond if edit_mode else None)
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
                wandb.log({'samples': wandb.Image(all_samples)}, step=train_state.step)
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        ############ save checkpoint and evaluate results
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')

            if accelerator.local_process_index == 0:
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
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



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
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
