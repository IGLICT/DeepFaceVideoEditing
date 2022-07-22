import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import imageio
import numpy as np
import scipy.interpolate
import jittor as jt
from tqdm import tqdm
from networks_stylegan3 import Generator
from utils import save_img, EasyDict

import legacy

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = jt.clamp(img * 127.5 + 128, 0, 255).astype(jt.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.detach().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, device=None, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = jt.array(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds]))
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].detach().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)
    
    
    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = jt.array(interp(frame_idx / w_frames))
                start = time.time()
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]
                end = time.time()
                #print(end - start)
                imgs.append(img.detach())
                #imgs.append(img)

        video_out.append_data(layout_grid(jt.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
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

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str
):
    """Render a latent vector interpolation video.

    Examples:
    #python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 --network=./weights/jt_stylegan3_ffhq_weights_t.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)

    kwargs = EasyDict(
            z_dim                   = 512,
            c_dim                   = 0,
            w_dim                   = 512,
            img_resolution          = 1024,
            img_channels            = 3,
        )
    
    print(kwargs)
    G = Generator(**kwargs)
    weight_dict = jt.load(network_pkl)
    G.load_state_dict(weight_dict)
    gen_interp_video(G=G, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

