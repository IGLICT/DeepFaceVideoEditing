import os
import jittor as jt
import numpy as np

jt.flags.use_cuda = 1
#----------------------------------------------------------------------------

def code_op_read_file():
    # source_filename = name + '.cc'
    with open("./models/stylegan3/ops/upfirdn2d.h", 'r', encoding='utf-8') as f:
        header_content = f.read()
    # with open(os.path.join(module_path, source_filename), 'r', encoding='utf-8') as f:
    #     source_content = f.read()
    return header_content, ""

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh

class Upfirdn2d(jt.Function):
    def __init__(self, up=1, down=1, padding=0, gain=1., flip_filter=False):
        assert isinstance(up, int) and up >= 1
        assert isinstance(down, int) and down >= 1
        self.up = up
        self.down = down
        self.upx, self.upy = _parse_scaling(up)
        self.downx, self.downy = _parse_scaling(down)
        self.px0, self.px1, self.py0, self.py1 = _parse_padding(padding)
        assert gain == float(gain) and gain > 0
        gain = float(gain)
        self.gain = gain
        self.x_shape = None
        self.y_shape = None
        self.save_f = None
        self.save_output = None
        self.flip_filter = flip_filter

    def single_upfird2d(self, x, f, cuda_header, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain, output_shape=()):
        cuda_src = f'''
            @alias(x_inp, in0)
            @alias(f_inp, in1)
            @alias(y_oup, out0)
            cudaStream_t stream = 0;
            assert(f_inp_shape0 >= 1 && f_inp_shape1 >= 1);
            int downx = {downx};
            int downy = {downy};
            int upx = {upx};
            int upy = {upy};
            int padx0 = {padx0};
            int padx1 = {padx1};
            int pady0 = {pady0};
            int pady1 = {pady1};
            float gain = {gain};
            bool flip = {1 if flip_filter else 0};
            assert(upx >= 1 && upy >= 1);
            assert(downx >= 1 && downy >= 1);

            // Create output tensor.
            int outW = ((int)x_inp_shape3 * upx + padx0 + padx1 - (int)f_inp_shape1 + downx) / downx;
            int outH = ((int)x_inp_shape2 * upy + pady0 + pady1 - (int)f_inp_shape0 + downy) / downy;
            assert(outW >= 1 && outH >= 1);

            // Initialize CUDA kernel parameters.
            upfirdn2d_kernel_params p;
            p.x             = x_inp_p;
            p.f             = f_inp_p;
            p.y             = y_oup_p;
            p.up            = make_int2(upx, upy);
            p.down          = make_int2(downx, downy);
            p.pad0          = make_int2(padx0, pady0);
            p.flip          = (flip) ? 1 : 0;
            p.gain          = gain;
            p.inSize        = make_int4((int)x_inp_shape3, (int)x_inp_shape2, (int)x_inp_shape1, (int)x_inp_shape0);
            p.inStride      = make_int4((int)x_inp_stride3, (int)x_inp_stride2, (int)x_inp_stride1, (int)x_inp_stride0);
            p.filterSize    = make_int2((int)f_inp_shape1, (int)f_inp_shape0);
            p.filterStride  = make_int2((int)f_inp_stride1, (int)f_inp_stride0);
            p.outSize       = make_int4((int)y_oup_shape3, (int)y_oup_shape2, (int)y_oup_shape1, (int)y_oup_shape0);
            p.outStride     = make_int4((int)y_oup_stride3, (int)y_oup_stride2, (int)y_oup_stride1, (int)y_oup_stride0);
            p.sizeMajor     = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
            p.sizeMinor     = (p.inStride.z == 1) ? p.inSize.z : 1;

            // Choose CUDA kernel.
            upfirdn2d_kernel_spec spec;
            spec = choose_upfirdn2d_kernel<float32>(p);

            // Set looping options.
            p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
            p.loopMinor     = spec.loopMinor;
            p.loopX         = spec.loopX;
            p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
            p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;

            // Compute grid size.
            dim3 blockSize, gridSize;
            if (spec.tileOutW < 0) // large
            {{
                blockSize = dim3(4, 32, 1);
                gridSize = dim3(
                    ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
                    (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
                    p.launchMajor);
            }}
            else // small
            {{
                blockSize = dim3(256, 1, 1);
                gridSize = dim3(
                    ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
                    (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
                    p.launchMajor);
            }}

            // Launch CUDA kernel.
            void* args[] = {{&p}};
            CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, stream));
        '''
        if len(output_shape) == 0:
            outW = (x.shape[3] * upx + padx0 + padx1 - f.shape[1] + downx) // downx
            outH = (x.shape[2] * upy + pady0 + pady1 - f.shape[0] + downy) // downy
            self.y_shape = (x.shape[0], x.shape[1], outH, outW)
            output = jt.code(self.y_shape, x.dtype, [x, f], cuda_header=cuda_header, cuda_src=cuda_src)
        else:
            output = jt.code(output_shape, x.dtype, [x, f], cuda_header=cuda_header, cuda_src=cuda_src)
        return output

    def execute(self, x, f): 
        assert x.dtype == jt.float32, "Only support float32 for now."
        if f is None:
            f = jt.ones([1, 1])
        if f.ndim == 1 and f.shape[0] == 1:
            f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
        assert f.dtype == jt.float32, "Only support float32 for now."
        cuda_header = code_op_read_file()[0]
        self.save_f = f
        if f.ndim == 2:
            output = self.single_upfird2d(x, f, cuda_header, self.upx, self.upy, self.downx, self.downy, self.px0, self.px1, self.py0, self.py1, self.flip_filter, self.gain)
        else:
            output = self.single_upfird2d(x, f.unsqueeze(0), cuda_header, self.upx, 1, self.downx, 1, self.px0, self.px1, 0, 0, self.flip_filter, 1.0)
            output = self.single_upfird2d(output, f.unsqueeze(1), cuda_header, 1, self.upy, 1, self.downy, 0, 0, self.py0, self.py1, self.flip_filter, self.gain)
        self.x_shape = x.shape
        self.x_dtype = x.dtype
        return output

    def grad(self, grads):  # only support backward for bias and input now (in pytorch.)
        cuda_header = code_op_read_file()[0]
        # print(grads)
        _, _, ih, iw = self.x_shape
        _, _, oh, ow = self.y_shape
        fw, fh = _get_filter_size(self.save_f)
        p = [
            fw - self.px0 - 1,
            iw * self.upx - ow * self.downx + self.px0 - self.upx + 1,
            fh - self.py0 - 1,
            ih * self.upy - oh * self.downy + self.py0 - self.upy + 1,
        ]
        if self.save_f.ndim == 2:
            dx = self.single_upfird2d(grads, self.save_f, cuda_header, self.downx, self.downy, self.upx, self.upy, p[0], p[1], p[2], p[3], not self.flip_filter, self.gain, self.x_shape)
        else:
            output = self.single_upfird2d(grads, self.save_f.unsqueeze(0), cuda_header, self.downx, 1, self.upx, 1, p[0], p[1], 0, 0, not self.flip_filter, 1.0)
            dx = self.single_upfird2d(output, self.save_f.unsqueeze(1), cuda_header, 1, self.downy, 1, self.upy, 0, 0, p[2], p[3], not self.flip_filter, self.gain, self.x_shape)
        return dx, None

if __name__ == "__main__":
    nx = np.random.uniform(0,1,(1,1,6,6)).astype("float32")
    nf = np.random.uniform(-1,1,(3,)).astype("float32")
    frelu = Upfirdn2d()
    x = jt.array(nx)
    f = jt.array(nf)
    out = frelu(x ,f)
    print(out)
    # fd = jt.array(nfd)
    # bias = jt.array(nbias)
    # up = 2
    # down = 2
    # frelu = Filtered_LReLU(fu=fu, fd=fd, up=up, down=down)
    # out = frelu(x, bias)
    # # print(out)
    grad = jt.grad(out, x)
    print(grad)