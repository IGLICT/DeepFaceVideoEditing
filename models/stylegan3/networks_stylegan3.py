import jittor as jt
import numpy as np
from .ops.filter_lrelu import Filtered_LReLU
import scipy.signal
import scipy.optimize
from typing import Any

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    frelu = Filtered_LReLU(fu=fu, fd=fd, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)
    out = frelu(x, b)
    return out

def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return jt.nn.conv2d(x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    #misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    #misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    #misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * (1.0 / (w*w).mean([1,2,3], keepdims=True).sqrt())
        s = s * (1.0 / (s*s).mean().sqrt())

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = 1.0 / ((w*w).sum(dims=[2,3,4]) + 1e-8).sqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d(x=x, weight=w.astype(x.dtype), padding=padding, groups=batch_size)
    
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

class FullyConnectedLayer(jt.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = jt.randn([out_features, in_features]) * (weight_init / lr_multiplier)
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        #self.bias = jt.float32(bias_init / lr_multiplier) if bias else None
        self.bias = jt.array(bias_init / lr_multiplier) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def execute(self, x):
        w = self.weight.astype(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.astype(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        
        if self.activation == 'linear' and b is not None:
            x = jt.matmul(x, w.t()) + b.unsqueeze(0)
        elif self.activation == 'lrelu':
            x = x.matmul(w.t()) + b.unsqueeze(0)
            x = jt.nn.leaky_relu(x)
        else:
            print("no implementation")
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class MappingNetwork(jt.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.w_avg = jt.zeros([w_dim])
        self.w_avg.requires_grad = False

    def execute(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        #misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.astype(jt.float32)
        x = x * (1.0 / ((x*x).mean(1, keepdims=True) + 1e-8).sqrt())
        if self.c_dim > 0:
            #misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.astype(jt.float32))
            y = y * (1.0 / (y.square().mean(1, keepdim=True) + 1e-8).sqrt())
            x = jt.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        #do not inplement it
        #if update_emas:
        #    self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            #x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
            x[:, :truncation_cutoff] = self.w_avg + truncation_psi * (x[:, :truncation_cutoff] - self.w_avg)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

class SynthesisInput(jt.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = jt.randn([self.channels, 2])
        radii = (freqs*freqs).sum(dim=1, keepdims=True).sqrt()
        freqs /= radii * (radii*radii).exp().pow(0.25)
        freqs *= bandwidth
        phases = jt.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = jt.randn([self.channels, self.channels])
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        #self.register_buffer('transform', jt.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        #self.register_buffer('freqs', freqs)
        #self.register_buffer('phases', phases)
        self.transform = jt.init.eye((3, 3))
        self.freqs = freqs
        self.phases = phases
        self.transform.requires_grad = False
        self.freqs.requires_grad = False
        self.phases.requires_grad = False
        
    def execute(self, w):
        # Introduce batch dimension.
        transforms = jt.array(self.transform).unsqueeze(0) # [batch, row, col]
        #print(transforms)
        freqs = jt.array(self.freqs).unsqueeze(0) # [batch, channel, xy]
        phases = jt.array(self.phases).unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = jt.array(jt.init.eye(3)).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        #print(m_r)
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = jt.array(jt.init.eye(3)).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        #print(m_t)
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        #print(transforms)
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = jt.array(jt.init.eye((2, 3)))
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate

        grids = jt.nn.affine_grid(theta.unsqueeze(0), [1, 1, int(self.size[1]), int(self.size[0])], align_corners=False)

        # Compute Fourier features.
        freqs = freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)
        freqs = freqs.repeat(1, 36, 36, 1, 1)
        x = (grids.unsqueeze(3) @ freqs).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = jt.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        #misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

class SynthesisLayer(jt.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        #3 original; Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,     # ######## Original:False Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
    ):
        super().__init__()
        #this version do not support fp16
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = jt.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel])
        self.bias = jt.zeros([self.out_channels])
        self.magnitude_ema = jt.ones([])
        self.magnitude_ema.requires_grad = False

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        #self.register_buffer('up_filter', self.design_lowpass_filter(
        #    numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))
        self.up_filter = self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate)
        if self.up_filter is not None:
            self.up_filter.requires_grad = False

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        #self.register_buffer('down_filter', self.design_lowpass_filter(
        #    numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))
        self.down_filter = self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial)
        if self.down_filter is not None:
            self.down_filter.requires_grad = False

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def execute(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        assert noise_mode in ['random', 'const', 'none'] # unused
        #misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        #misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        #if update_emas:
        #    with torch.autograd.profiler.record_function('update_magnitude_ema'):
        #        magnitude_cur = x.detach().to(torch.float32).square().mean()
        #        self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = 1.0 / self.magnitude_ema.sqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        #dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        dtype = jt.float32
        x = modulated_conv2d(x=x.astype(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        #x = filter_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.astype(x.dtype),
        x = filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.astype(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        #misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        #assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return jt.float32(f)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return jt.float32(f)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

class SynthesisNetwork(jt.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        #print("SynthesisInput")

        # Construct layers.
        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])

        #print("Synthesis Layer")

        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            #use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            use_fp16 = False
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels[prev]), out_channels= int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def execute(self, ws, **layer_kwargs):
        #misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = jt.misc.unbind(ws, dim=1)

        # Execute layers.
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        #misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.astype(jt.float32)
        return x
    
    def execute_lists(self, ws, **layer_kwargs):
        ws = jt.misc.unbind(ws, dim=1)

        # Execute layers.
        x_list = []
        x = self.input(ws[0])
        x_list.append(x)
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
            x_list.append(x)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        #misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.astype(jt.float32)
        return x_list, x

    def execute_mask(self, ws, mask_list, x_edit_list, start_list, end_list, **layer_kwargs):
        #misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        #ws = ws.to(torch.float32).unbind(dim=1)
        ws = jt.misc.unbind(ws, dim=1)
        
        pad = jt.nn.ZeroPad2d(padding=(10, 10, 10, 10))
        ###########[0   1   2   3   4   5   6    7    8    9    10   11    12    13    #14   #15 ]
        res_list = [16, 16, 16, 32, 32, 64, 128, 128, 256, 256, 512, 1024, 1024, 1024, 1024, 1024]

        # Execute layers.
        x = self.input(ws[0])
        count = 0
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
            count += 1
            if count >= start_list and count < end_list:
                res = res_list[count]
                #print(res)
                for edit_num in range(len(x_edit_list)):
                    mask_res = jt.nn.interpolate(mask_list[edit_num], size=(res, res), mode='bilinear')
                    mask_res = pad(mask_res)
                    x = x_edit_list[edit_num][count] * mask_res + x * (1 - mask_res)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        #misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.astype(jt.float32)
        return x

class Generator(jt.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def execute(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img


