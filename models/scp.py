import jittor as jt
from jittor import nn
from models.encoders import psp_encoders
from models.stylegan3.networks_stylegan3 import Generator
import numpy as np

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name and k[len(name)] == '.'}
    return d_filt

class Linear_block(nn.Module):
    def __init__(self):
        super(Linear_block, self).__init__()
        modules = []
        modules += [nn.Linear(1024 + 512, 512),
                    nn.LeakyReLU()]
        num_pools = 2
        for i in range(num_pools-1):
        	modules += [nn.Linear(512, 512),
                    nn.LeakyReLU()]
        modules += [nn.Linear(512, 512)]
        self.linear = nn.Sequential(*modules)

    def execute(self, x):
        return self.linear(x)

class scp(nn.Module):
    def __init__(self, opts):
        super(scp, self).__init__()
        self.opts = opts
        # Define architecture
        self.img_encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        self.sketch_encoder = psp_encoders.GradualStyleEncoder(1, 50, 'ir_se', self.opts)
        self.mask_encoder = psp_encoders.GradualStyleEncoder(1, 50, 'ir_se', self.opts)
        for layer_index in range(0, 16):
            block = Linear_block()
            setattr(self, f'b{layer_index}', block)
        self.decoder = Generator(z_dim=512, c_dim=0, w_dim=512,img_resolution=1024, img_channels=3)
        self.face_pool = jt.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = jt.load(self.opts.checkpoint_path)
            self.img_encoder.load_state_dict(get_keys(ckpt, 'img_encoder'))
            self.sketch_encoder.load_state_dict(get_keys(ckpt, 'sketch_encoder'))
            self.mask_encoder.load_state_dict(get_keys(ckpt, 'mask_encoder'))
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'))
            for layer_index in range(16):
                block = getattr(self, f'b{layer_index}')
                layer_ckpt = get_keys(ckpt, f'b{layer_index}')
                block.load_state_dict(layer_ckpt)
        else:
            print("Don't provide scp network weights")

    def execute(self, img_input, sketch, mask, resize=True):
        #Encode input images, mask, sketch
        img_code = self.img_encoder(img_input)
        sketch_code = self.sketch_encoder(sketch)
        mask_code = self.mask_encoder(mask)
        #print(img_code.shape)
        #print(sketch_code.shape)
        #print(mask_code.shape)

        #Fuse the results
        add_code = jt.concat((img_code, sketch_code, mask_code), dim=2)
        fusion_code = []
        for layer_index in range(16):
            code = add_code[:, layer_index, :]
            block = getattr(self, f'b{layer_index}')
            code = block(code)
            fusion_code.append(code)
        fusion_code = jt.misc.stack(fusion_code, dim=1)
        images = self.decoder.synthesis(fusion_code, noise_mode='const', force_fp32=True)

        if resize:
            images = self.face_pool(images)

        return images, fusion_code, img_code


