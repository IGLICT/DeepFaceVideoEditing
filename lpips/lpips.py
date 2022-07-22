import jittor as jt
import jittor.nn as nn

import lpips.pretrained_networks as pn

def spatial_average(in_tens, keepdims=True):
    return in_tens.mean([2,3],keepdims=keepdims)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = jt.float32([-.030,-.088,-.188])[None,:,None,None]
        self.scale = jt.float32([.458,.448,.450])[None,:,None,None]
        self.shift.requires_grad = False
        self.scale.requires_grad = False

    def execute(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = jt.sqrt(jt.sum(in_feat**2,dim=1,keepdims=True))
    return in_feat/(norm_factor+eps)

# Learned perceptual metric
class LPIPS(jt.nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss jittor.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()
        
        if(version == 'v0.0'):
            print("not implement in Jittor!")

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    model_path = './lpips/weights/jt_lpips_%s_v01.pkl'%net
                if(verbose):
                    print('Loading model from: %s'%model_path)
                model_dict = jt.load(model_path)
                self.load_state_dict(model_dict)

        if(eval_mode):
            self.eval()

    def execute(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.execute(in0_input), self.net.execute(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            #feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdims=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdims=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdims=True), keepdims=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

if __name__ == "__main__":
    #model_dict = jt.load('./checkpoints/jt_lpips_alex_v01.pkl')

    loss_fn = LPIPS(net='alex', spatial=False)
    #jt.save(loss_fn.state_dict(), './checkpoints/jt_lpips_alex_v01.pkl')

    import jittor.transform as transform
    from PIL import Image
    img_size = 64
    transform_image = transform.Compose([
            transform.Resize(size = img_size),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def read_img(path):
        img = Image.open(path).convert('RGB')
        img = transform_image(img)
        img = jt.array(img)
        img = img.unsqueeze(0)
        return img
    ex_ref = read_img('./PerceptualSimilarity-master/imgs/ex_ref.png')
    ex_p0 = read_img('./PerceptualSimilarity-master/imgs/ex_p0.png')
    ex_p1 = read_img('./PerceptualSimilarity-master/imgs/ex_p1.png')

    ex_d0 = loss_fn.execute(ex_ref,ex_p0)
    ex_d1 = loss_fn.execute(ex_ref,ex_p1)

    print('Distances: (%.8f, %.8f)'%(ex_d0, ex_d1))
    
