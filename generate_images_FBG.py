# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

#
# This template file is designed to work with the EDM2 github repository: https://github.com/NVlabs/edm2/tree/main
#     It is a placeholder of the "generate_images.py" file in the original repository
#
#     It allows for both stochastic sampling and following the PFODE using either 1st order Euler or 2nd order Heun sampling
#     It allows for testing the three guidance schemes desribed in the paper: Classifier-Free Guidance (CFG), Limited Interval Guidance (LIG) and Feedback Guidance (FBG) 
#               as well as two Hybrid methods that combine FBG with CFG and LIG.
#
# 

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from matplotlib import pyplot as plt
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

def set_seed(seed: int) -> None:
    ''' Function to set the seed for all pseudorandom number generators for reproducible results, even in the context of stochastic sampling

    Input:
        - seed : The desired seed
    Output:
        - None
    '''
    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For some libraries that use environment variables for seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensuring reproducibility of operations on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#----------------------------------------------------------------------------
# Configuration presets.
#   We use the default FID optimised networks from the EDM2 repository. 
#   We also do not use Autoguidance, i.e. we use the same size/training stage of networks for conditional and unconditional generation.
#      It should be noted that our framework is easily rewrittable to arrive at guidace equations similar to the ones proposed in Autoguidance.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
config_presets = {
    'edm2-img512-xs':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl'),
    'edm2-img512-s':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',    gnet=f'{model_root}/edm2-img512-s-2147483-0.025.pkl'),
    'edm2-img512-m':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',    gnet=f'{model_root}/edm2-img512-m-2147483-0.030.pkl'),
    'edm2-img512-l':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',    gnet=f'{model_root}/edm2-img512-l-1879048-0.015.pkl'), 
    'edm2-img512-xl':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',   gnet=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl'), 
    'edm2-img512-xxl':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl'), 
}

#----------------------------------------------------------------------------
# Sampler based on the EDM sampler from the paper (for simplicity we remove the  term in the PFODE methods)
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# Further extended to support CFG, LIG, FBG and Hybrid methods under  sampling, 1st order Euler or 2nd order Heun

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=64, sigma_min=0.002, sigma_max=80, rho=7, 
    # Added hyperparameters
    guidance_type = 'CFG', sampling_type = 'stochastic',
    constant_guidance=1., t_start=2.9, t_end=0.5,
    temp=0., offset=0., pi=0.95, t_0 = 0.5, t_1 = 0.4, max_guidance=10.,          
    dtype=torch.float32, randn_like=torch.randn_like,
    # For visualisation/tracking
    print_guids = False,
):
    assert guidance_type in ['CFG','LIG','FBG','Hybrid_CFG_FBG', 'Hybrid_LIG_FBG']
    assert sampling_type in ['stochastic','1st_order_Euler','2nd_order_Heun']
    
    # From max guidscale to posterior ratio values
    minimal_log_posterior = np.log((1-pi)*max_guidance/(max_guidance-1))
    # If a Hybrid method is used one needs to compensate for the constant guidance applied: max_guidance-> max_guidance - (constant_guidance-1)
    if guidance_type in ['Hybrid_CFG_FBG','Hybrid_LIG_FBG']:   
        minimal_log_posterior = np.log((1-pi)*(max_guidance-constant_guidance+1)/(max_guidance-constant_guidance))

    print('Minimum log posterior value: ', minimal_log_posterior)
    
    # Guided denoiser.
    def denoise(x, t):
        cond_Dx = net(x, t, labels).to(dtype)
        uncond_Dx = gnet(x, t).to(dtype)
        return uncond_Dx, cond_Dx

    # Posterior ratio estimation
    def update_log_posterior(prev_log_posterior, x_cur, x_next, t_cur, t_next, uncond_Dx, cond_Dx):  
        # Compute mu's and sigma_{t-1|t}^2
        sigma_square_tilde_t  = (t_cur**2-t_next**2)*t_next**2/t_cur**2                                 # Backward transition kernel variance $$\sigma_{t-1|t}^2$$
        uncond_predicted_mean = t_next**2/t_cur**2 * x_cur + (t_cur**2-t_next**2)/t_cur**2 * uncond_Dx  # Predicted unconditional mean at t   $$\mu_c(x_{t}|c)$$
        cond_predicted_mean   = t_next**2/t_cur**2 * x_cur + (t_cur**2-t_next**2)/t_cur**2 * cond_Dx    # Predicted conditional mean at t     $$\mu(x_{t})$$
        predicted_noised_x    = x_next                                                                  # noisy state x_{t-1}
    
        # Compute error difference
        cond_MSE   = torch.sum((predicted_noised_x - cond_predicted_mean)**2  ,dim=[1,2,3])             # MSE between x_t and mu_c,t+1
        uncond_MSE = torch.sum((predicted_noised_x - uncond_predicted_mean)**2,dim=[1,2,3])             # MSE between x_t and mu_t+1
        diff = cond_MSE-uncond_MSE                                                                      # Difference between the two terms

        # Update the log posterior ratio
        log_posterior = prev_log_posterior -temp/(2*sigma_square_tilde_t)*diff + offset                 # Update the log posterior ratio
        log_posterior = torch.clamp(log_posterior ,min=minimal_log_posterior, max=3.)                   # Clamp the log posterior ratio (to avoid negative guidance scales)
        return log_posterior

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    # Initialize the state and log poterior values
    x_next = noise.to(dtype) * t_steps[0]
    log_posterior  = torch.zeros(x_next.size(0), device=noise.device)          
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):      # 0, ..., N-1
        # Compute the guidance scale for FBG (ignored in case of LIG or CFG)
        guidance_scale = torch.exp(log_posterior)/(torch.exp(log_posterior)-(1-pi))
        # In case of hybrid methods: add the constant guidance specified in the relevant intervals
        if guidance_type == 'Hybrid_CFG_FBG': 
            guidance_scale += constant_guidance-1
        if guidance_type == 'Hybrid_LIG_FBG' and t_cur<t_start and t_cur>t_end:
            guidance_scale += constant_guidance-1
        if print_guids and guidance_type in ['FBG','Hybrid_CFG_FBG','Hybrid_LIG_FBG']:
            print(f'Guidance scale {i}: ', guidance_scale)

        # Step
        x_cur = x_next

        # Compute the unconditional and conditional scores
        uncond_Dx, cond_Dx = denoise(x_cur, t_cur)

        # Mix the scores to obtain the desired *guided* score
        if guidance_type == 'CFG':
            guided_Dx = uncond_Dx + constant_guidance*(cond_Dx - uncond_Dx)
            if print_guids:
                print(f'Guidance scale {i}: ', constant_guidance)
        if guidance_type == 'LIG':
            constant_guidance_t = torch.where((t_cur<t_start) & (t_cur>t_end),constant_guidance,1.)
            guided_Dx = uncond_Dx + constant_guidance_t*(cond_Dx - uncond_Dx)
            if print_guids:
                print(f'Guidance scale {i}: ', constant_guidance_t)
        if guidance_type in ['FBG', 'Hybrid_CFG_FBG','Hybrid_LIG_FBG']:
            guided_Dx = uncond_Dx + guidance_scale[:,None,None,None]*(cond_Dx - uncond_Dx)

        # Apply sampling step along using the desired sampler
        if sampling_type == 'stochastic':
            x_next = t_next**2/t_cur**2 * x_cur + (t_cur**2-t_next**2)/t_cur**2 * guided_Dx 
            if i < num_steps - 1:
                pure_stochastic_noise = torch.randn_like(x_next)
                std_for_stochastic_noise = torch.sqrt(t_cur**2-t_next**2) * t_next/t_cur
                x_next = x_next + std_for_stochastic_noise * pure_stochastic_noise
        if sampling_type == '1st_order_Euler':
            x_next = x_cur + (t_next-t_cur)  * (x_cur-guided_Dx)/t_cur 
        if sampling_type == '2nd_order_Heun':
            x_next = x_cur + (t_next-t_cur)  * (x_cur-guided_Dx)/t_cur 
            # Apply 2nd order correction.
            if i < num_steps - 1:
                uncond_Dx_next, cond_Dx_next = denoise(x_next, t_next)
                if guidance_type == 'CFG':
                    guided_Dx_next = uncond_Dx_next + constant_guidance*(cond_Dx_next - uncond_Dx_next)
                if guidance_type == 'LIG':
                    constant_guidance_t = torch.where((t_cur<t_start) & (t_cur>t_end),constant_guidance,1.)
                    guided_Dx_next = uncond_Dx_next + constant_guidance_t*(cond_Dx_next - uncond_Dx_next)
                if guidance_type in ['FBG', 'Hybrid_CFG_FBG','Hybrid_LIG_FBG']:                               # Notice here we use the same guidance scale as atprevious timestep (adapting this is left as future work)
                    guided_Dx_next = uncond_Dx_next + guidance_scale[:,None,None,None]*(cond_Dx_next - uncond_Dx_next)       
                x_next = x_cur + (t_next - t_cur) * (0.5 * (x_cur-guided_Dx)/t_cur + 0.5 * (x_next-guided_Dx_next)/t_next)

        # Update the log posterior value that is used to compute the guidance scale in the next timestep
        log_posterior = update_log_posterior(log_posterior, x_cur, x_next, t_cur, t_next, uncond_Dx, cond_Dx)
        
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 256,                  # Maximum batch size for the diffusion model. (32)
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,     # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    #seeds = torch.linspace(0,3,4).long()
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Make sure output directory exists
            if outdir is not None:
                os.makedirs(outdir, exist_ok=True)

            # The function to compute the offset parameter from t0 and pi
            def delta_from_t0(N_steps, pi, t0, sigma_square_tilde, lambda_ref = 3.):
                if verbose:
                    print(f'(Info) t0 is at timestep {round((N_steps-1)*t0,3)} of {N_steps-1}')
                # Compute the Offset parameter
                delta = 1/((1-t0)*N_steps)*torch.log(torch.tensor((1-pi)*lambda_ref/(lambda_ref-1)))
                return round(delta.item(),4)

            # The function to compute the temperature parameter from the offset parameter and t1
            def Temp_from_t1(N_steps, delta, t1, sigma_square_tilde, alpha=10.):
                if verbose:
                    print(f'(Info) t1 is at timestep {round((N_steps-1)*t1,3)} of {N_steps-1}')
                t1_lower = int(torch.floor(torch.tensor(t1*N_steps)).item())
                # In case t1 is not located at a particular transition timestep => take the linear interpolation of the transition variance surrounding t1 
                a = t1*N_steps - t1_lower                                                                            
                sigma_square_tilde_t1, sigma_square_tilde_t1_next = sigma_square_tilde[t1_lower], sigma_square_tilde[t1_lower+1]
                sigma_square_tilde = (1-a)*sigma_square_tilde_t1+a*sigma_square_tilde_t1_next
                # Compute the temperature
                temp = torch.abs(2*sigma_square_tilde/alpha * delta)
                return round(temp.item(),4)

            # The function to compute the discrete transition kernel variances used during sampling
            def get_sigma_square_tilde(num_steps, rho, sigma_min, sigma_max):
                sigma_square_tilde = torch.zeros(num_steps)
                step_indices = torch.arange(num_steps)
                t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    sigma_square_tilde[num_steps-i-1] = (t_cur**2-t_next**2) * t_next**2/t_cur**2
                return sigma_square_tilde
                
            if sampler_kwargs['temp']==-1. and sampler_kwargs['guidance_type'] in ['FBG', 'Hybrid_CFG_FBG', 'Hybrid_LIG_FBG']:
                print('Computing $delta$, $tau$ from $pi$, $t_0$ and $t_1$ ...\n    (if not desired specify --Offset and --Temp directly)\n')
                sigma_square_tilde = get_sigma_square_tilde(sampler_kwargs['num_steps'],sampler_kwargs['rho'],sampler_kwargs['sigma_min'],sampler_kwargs['sigma_max'])
                sampler_kwargs['offset'] = delta_from_t0(sampler_kwargs['num_steps'], sampler_kwargs['pi'], sampler_kwargs['t_0'], sigma_square_tilde, lambda_ref = 3.)
                sampler_kwargs['temp'] = Temp_from_t1(sampler_kwargs['num_steps'], sampler_kwargs['offset'], sampler_kwargs['t_1'], sigma_square_tilde, alpha=10.)

            if verbose and sampler_kwargs['guidance_type'] in ['FBG', 'Hybrid_CFG_FBG', 'Hybrid_LIG_FBG']:
                print(r'(Info) Pi value $\pi$:         ', sampler_kwargs['pi'])
                print(r'(Info) offset value $\delta$ (should be negative):  ', sampler_kwargs['offset'])
                print(r'(Info) temp value $\tau$ (should be positive):       ', sampler_kwargs['temp'])
            if verbose and sampler_kwargs['guidance_type']=='CFG':
                print(r'(Info) Guidance scale value $\lambda$: ', sampler_kwargs['constant_guidance'])
            if verbose and sampler_kwargs['guidance_type']=='LIG':
                print(r'(Info) Guidance scale value $\lambda$: ', sampler_kwargs['constant_guidance'])
                print(r'(Info) Starting point of guidance:     ', sampler_kwargs['t_start'])
                print(r'(Info) Ending point of guidance:       ', sampler_kwargs['t_end'])

            if verbose:
                print('\n(Info) Number of steps (should be halved for fair comparison using 2nd_order_Heun sampling): ', sampler_kwargs['num_steps'])
            
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                #set the seed manually!
                set_seed(seeds[batch_idx])

                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)

                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
# General hyperparameters
@click.option('--preset',                                 help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                                    help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--gnet',                                   help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                                 help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                                help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                                  help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',                     help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',                help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=256, show_default=True)

# Scheduler hyperparameters
@click.option('--steps', 'num_steps',                     help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--sigma_min',                              help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                              help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                                    help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)

# Choose sampling type
@click.option('--guidance_type', 'guidance_type',         help='Type of guidance used: CFG, LIG, FBG, Hybrid_CFG_FBG, Hybrid_LIG_FBG', metavar='STR',     type=str, default='CFG', required=True)
@click.option('--sampling_type', 'sampling_type',         help='Type of sampling used: stochastic, 1st_order_Euler, 2nd_order_Heun',   metavar='STR',     type=str, default='stochastic', required=True)
@click.option('--print_guidance_scales', 'print_guids',   help='Should the guidance scale be printed throughout inference?',  is_flag=True) # Deactivated by default


# Guidance scheme specific hyperparameters
#     For CFG
@click.option('--constant_guidance', 'constant_guidance', help='Maximal guidance scale value', metavar='FLOAT',                   type=click.FloatRange(min=0), default=2)
#     For LIG
@click.option('--t_start', 't_start',                     help='Starting point for Limited interval guidance', metavar='FLOAT',   type=click.FloatRange(min=0), default=1.6)
@click.option('--t_end', 't_end',                         help='Ending point for Limited interval guidance', metavar='FLOAT',     type=click.FloatRange(min=0), default=0.28) # CHECK THIS!!
#     For FBG
@click.option('--pi', 'pi',                               help='Pi param. for the guidance stremgth', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.5)
@click.option('--t_0', 't_0',                             help='t_0 param. for posterior estimation', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.5)
@click.option('--t_1', 't_1',                             help='t_1 param. for posterior estimation', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.35)
@click.option('--temp', 'temp',                           help='Temperature param. $\tau$ for posterior estimation', metavar='FLOAT',    type=click.FloatRange(min=-1), default=-1.)
@click.option('--offset', 'offset',                       help='Offset param. $\delta$for posterior estimation', metavar='FLOAT',        type=click.FloatRange(min=-1), default=-1.)
@click.option('--max_guidance', 'max_guidance',           help='Maximal guidance scale value', metavar='FLOAT',                          type=click.FloatRange(min=1), default=2)



def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException("Please specify valid preset: --preset 'edm2-img512-(size)'")

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()
    torch.cuda.empty_cache()

#----------------------------------------------------------------------------
