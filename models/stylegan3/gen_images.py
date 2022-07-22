import os
import re
from typing import List, Optional, Tuple, Union

import click
import numpy as np
import jittor as jt
from networks_stylegan3 import Generator
from utils import save_img, EasyDict
from imageio import imread,imsave

import legacy

jt.flags.use_cuda = 1

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
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

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:
    python gen_images.py --outdir=out --trunc=1 --seeds=2 --network=./weights/jt_stylegan3_ffhq_weights_t.pkl

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

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = jt.zeros([1, G.c_dim])
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z = jt.array(np.random.RandomState(seed).randn(1, 512))

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            print(m)
            G.synthesis.input.transform = jt.array(m)
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

        image = img.squeeze(0).detach().numpy()
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        imsave(f'{outdir}/seed{seed:04d}.png',image)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------


