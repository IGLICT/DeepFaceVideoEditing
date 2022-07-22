import os
import jittor as jt
import numpy as np
from .upfird2d import Upfirdn2d
import time

jt.flags.use_cuda = 1
#----------------------------------------------------------------------------

def code_op_read_file():
    # source_filename = name + '.cc'
    with open("./models/stylegan3/ops/filter_lrelu.h", 'r', encoding='utf-8') as f:
        header_content = f.read()
    # with open(os.path.join(module_path, source_filename), 'r', encoding='utf-8') as f:
    #     source_content = f.read()
    return header_content, ""

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0] # width, height

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

def filter_lrelu_act(x, si, sx, sy, gain, slope, clamp, writeSigns, y_shape, so_shape):
    cuda_header = code_op_read_file()[0]
    s_shape3 = 0
    s_shape2 = 0
    # print(si)
    if si.numel() != 0: # si.ndim == 4:
        s_shape3 = si.shape[-1]
        s_shape2 = si.shape[2]
    cuda_src = f'''
        @alias(x_inp, out1)
        @alias(s_inp, out2)
        @alias(s_oup, out0)
        // Set CUDA device.
        // Validate arguments.
        cudaStream_t stream = 0;
        float gain = {gain};
        float slope = {slope};
        float clamp = {clamp};
        int sx = {sx};
        int sy = {sy};
        bool writeSigns = {1 if writeSigns else 0};
        // Output signs if we don't have sign input.
        uint8_t* s = s_inp_p;
        bool readSigns = !!{si.numel()};
        int s_shape3 = {s_shape3};
        int s_shape2 = {s_shape2};
        if (writeSigns)
        {{
            int64_t sw = x_inp_shape3;
            sw = (sw + 15) & ~15; // Round to a multiple of 16 for coalescing.
            s = s_oup_p;
            s_shape3 = s_oup_shape3;
            s_shape2 = s_oup_shape2;
        }}


        filtered_lrelu_act_kernel_params p;
        p.x         = x_inp_p;
        p.s         = (readSigns || writeSigns) ? s : 0;
        p.gain      = gain;
        p.slope     = slope;
        p.clamp     = clamp;
        p.xShape    = make_int4((int)x_inp_shape3, (int)x_inp_shape2, (int)x_inp_shape1, (int)x_inp_shape0);
        p.xStride   = make_longlong4(x_inp_stride3, x_inp_stride2, x_inp_stride1, x_inp_stride0);
        p.sShape    = (readSigns || writeSigns) ? make_int2((int)s_shape3 << 2, (int)s_shape2) : make_int2(0, 0); // Width is in elements. Contiguous.
        p.sOfs      = make_int2(sx, sy);

        // Choose CUDA kernel.
        void* func = 0;
        if (writeSigns)
            func = choose_filtered_lrelu_act_kernel<float, true, false>();
        else if (readSigns)
            func = choose_filtered_lrelu_act_kernel<float, false, true>();
        else
            func = choose_filtered_lrelu_act_kernel<float, false, false>();
        assert(func);

        // Launch CUDA kernel.
        void* args[] = {{&p}};
        int bx = 128; // 4 warps per block.

        // Logical size of launch = writeSigns ? p.s : p.x
        uint32_t gx = writeSigns ? p.sShape.x : p.xShape.x;
        uint32_t gy = writeSigns ? p.sShape.y : p.xShape.y;
        uint32_t gz = p.xShape.z * p.xShape.w; // Same as in p.sShape if signs are in use.
        gx = (gx - 1) / bx + 1;

        // Make sure grid y and z dimensions are within CUDA launch limits. Kernel loops internally to do the rest.
        const uint32_t gmax = 65535;
        gy = std::min(gy, gmax);
        gz = std::min(gz, gmax);

        // Launch.
        CUDA_CHECK(cudaLaunchKernel(func, dim3(gx, gy, gz), bx, args, 0, stream));
    '''
    so = jt.zeros(so_shape).uint8()
    so, x, si = jt.code([], [so, x, si], cuda_src=cuda_src, cuda_header=cuda_header)
    # jt.sync_all()
    return x, so

class Filtered_LReLU(jt.Function):
    def __init__(self, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, fu=None, fd=None, si=None, sx=0, sy=0):
        assert isinstance(up, int) and up >= 1
        assert isinstance(down, int) and down >= 1
        self.up = up
        self.down = down
        self.px0, self.px1, self.py0, self.py1 = _parse_padding(padding)
        assert gain == float(gain) and gain > 0
        gain = float(gain)
        self.gain = gain
        assert slope == float(slope) and slope >= 0
        slope = float(slope)
        self.slope = slope
        assert clamp is None or (clamp == float(clamp) and clamp >= 0)
        clamp = float(clamp if clamp is not None else 1 << 31)
        self.clamp = clamp
        #self.fu = fu
        #self.fd = fd
        #if self.fu is None:
        #    self.fu = jt.ones([1, 1])
        #if self.fd is None:
        #    self.fd = jt.ones([1, 1])         
        
        if fu is None:
            fu = jt.ones([1, 1])
        if fd is None:
            fd = jt.ones([1, 1])         
        self.fu = fu
        self.fd = fd

        assert 1 <= self.fu.ndim <= 2
        assert 1 <= self.fd.ndim <= 2
        if up == 1 and self.fu.ndim == 1 and self.fu.shape[0] == 1:
            fu = (fu ** 2).unsqueeze(0)
        if down == 1 and self.fd.ndim == 1 and self.fd.shape[0] == 1:
            fd = (fd ** 2).unsqueeze(0)
        self.si = si
        if self.si == None:
            self.si = jt.empty([0, 0, 0, 0]).uint8()
        self.sx = sx
        self.sy = sy
        self.x_shape = None
        self.x_dtype = None
        self.y_shape = None
        self.save_so = None
        self.writeSigns = None
        self.ret = 0
        self.flip_filter = flip_filter
        # TODO: add cache for kernels

    def execute(self, x, bias = None): 
        assert x.dtype == jt.float32, "Only support float32 for now."
        cuda_header = code_op_read_file()[0]
        if bias is None:
            bias = jt.zeros([x.shape[1]])
        self.save_bias = bias
        writeSigns = (self.si.numel() == 0) and (x.requires_grad or bias.requires_grad)
        self.writeSigns = 1 if writeSigns else 0
        if self.fu.ndim == 2:
            fushape_src = '''
            int fushape1 = fu_inp_shape1;
            int fushape0 = fu_inp_shape0;
            '''
            fustride_src = '''
            int fustride1 = fu_inp_stride1;
            int fustride0 = fu_inp_stride0;
            '''
        else:
            fushape_src = '''
            int fushape1 = fu_inp_shape0;
            int fushape0 = 0;
            '''
            fustride_src = '''
            int fustride1 = fu_inp_stride0;
            int fustride0 = 0;
            '''
        if self.fd.ndim == 2:
            fdshape_src = '''
            int fdshape1 = fd_inp_shape1;
            int fdshape0 = fd_inp_shape0;
            '''
            fdstride_src = '''
            int fdstride1 = fd_inp_stride1;
            int fdstride0 = fd_inp_stride0;
            '''
        else:
            fdshape_src = '''
            int fdshape1 = fd_inp_shape0;
            int fdshape0 = 0;
            '''
            fdstride_src = '''
            int fdstride1 = fd_inp_stride0;
            int fdstride0 = 0;
            '''
        cuda_src = f'''
            @alias(x_inp, in0)
            @alias(fu_inp, in1)
            @alias(fd_inp, in2)
            @alias(b_inp, in3)
            @alias(s_inp, in4)
            @alias(y_oup, out0)
            @alias(s_oup, out1)
            cudaStream_t stream = 0;
            int up = {self.up};
            int down = {self.down};
            int px0,py0,px1,py1;
            px0 = {self.px0};
            px1 = {self.px1};
            py0 = {self.py0};
            py1 = {self.py1};
            float gain = {self.gain};
            float slope = {self.slope};
            float clamp = {self.clamp};
            bool flip_filter = {1 if self.flip_filter else 0};
            bool writeSigns = {self.writeSigns};
            int sx = {self.sx};
            int sy = {self.sy};
            // Figure out how much shared memory is available on the device.
            int maxSharedBytes = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
            int sharedKB = maxSharedBytes >> 10;
            // Populate enough launch parameters to check if a CUDA kernel exists.
            filtered_lrelu_kernel_params p;
            p.up      = up;
            p.down    = down;
            {fushape_src}
            {fdshape_src}
            p.fuShape = make_int2(fushape1, fushape0); // shape [n, 0] indicates separable filter.
            p.fdShape = make_int2(fdshape1, fdshape0);
            filtered_lrelu_kernel_spec test_spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
            if (!test_spec.exec)
            {{
                // No kernel found - return empty tensors and indicate missing kernel with return code of -1.
                y_oup->set_shape({{0}});
                return;
            }} else {{
                // Input/output element size.
                // int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;
                int64_t sz = 4; // support float32 for now.
                // Input sizes.
                int64_t xw = (int)x_inp_shape3;
                int64_t xh = (int)x_inp_shape2;
                int64_t fut_w = fushape1 - 1;
                int64_t fut_h = (int)fu_inp_shape0 - 1;
                int64_t fdt_w = fdshape1 - 1;
                int64_t fdt_h = (int)fd_inp_shape0 - 1;
                // Logical size of upsampled buffer.
                int64_t cw = xw * up + (px0 + px1) - fut_w;
                int64_t ch = xh * up + (py0 + py1) - fut_h;
                // LOGir << "jittor shape" << fut_w << " " << fut_h << " " << fdt_w << " " << fdt_h << " " << cw << " " << ch;
                // assert(cw > fdt_w && ch > fdt_h, "upsampled buffer must be at least the size of downsampling filter");
                // assert(cw <= INT_MAX && ch <= INT_MAX, "upsampled buffer is too large");

                // Compute output size and allocate.
                int64_t yw = y_oup_shape3;
                int64_t yh = y_oup_shape2;
                // assert(yw > 0 && yh > 0, "output must be at least 1x1");
                // assert(yw <= INT_MAX && yh <= INT_MAX, "output is too large");

                // Allocate sign tensor.
                bool readSigns = !!{self.si.numel()};
                int64_t sw_active = 0; // Active width of sign tensor.
                uint8_t* so = s_inp_p;
                int s_shape3, s_shape2;
                if(!!{self.si.numel()}) {{
                    s_shape3 = s_inp_shape3;
                    s_shape2 = s_inp_shape2;
                }}
                if (writeSigns)
                {{
                    sw_active = yw * down - (down - 1) + fdt_w;     // Active width in elements.
                    int64_t sh = s_oup_shape2;    // Height = active height.
                    int64_t sw = s_oup_shape3;            // Width  = active width in elements, rounded up to multiple of 16.
                    // assert(sh <= INT_MAX && (sw >> 2) <= INT_MAX, "signs is too large");
                    so = s_oup_p;
                    s_shape3 = s_oup_shape3;
                    s_shape2 = s_oup_shape2;
                }}
                else if (readSigns)
                    sw_active = s_inp_shape3 << 2;
                // Populate rest of CUDA kernel parameters.
                p.x         = x_inp_p;
                p.y         = y_oup_p;
                p.b         = b_inp_p;
                p.s         = (readSigns || writeSigns) ? (unsigned char*)so : 0;
                p.fu        = (float*) fu_inp_p;
                p.fd        = (float*) fd_inp_p;
                p.pad0      = make_int2(px0, py0);
                p.gain      = gain;
                p.slope     = slope;
                p.clamp     = clamp;
                p.flip      = (flip_filter) ? 1 : 0;
                p.xShape    = make_int4((int)x_inp_shape3, (int)x_inp_shape2, (int)x_inp_shape1, (int)x_inp_shape0);
                p.yShape    = make_int4((int)y_oup_shape3, (int)y_oup_shape2, (int)y_oup_shape1, (int)y_oup_shape0);
                p.sShape    = (readSigns || writeSigns) ? make_int2((int)s_shape3, (int)s_shape2) : make_int2(0, 0); // Width is in bytes. Contiguous.
                p.sOfs      = make_int2(sx, sy);
                p.swLimit   = (sw_active + 3) >> 2; // Rounded up to bytes.

                // x, y, b strides are in bytes.
                p.xStride   = make_longlong4(sz * x_inp_stride3, sz * x_inp_stride2, sz * x_inp_stride1, sz * x_inp_stride0);
                p.yStride   = make_longlong4(sz * y_oup_stride3, sz * y_oup_stride2, sz * y_oup_stride1, sz * y_oup_stride0);
                p.bStride   = sz * b_inp_stride0;

                // fu, fd strides are in elements.
                
                {fustride_src}
                {fdstride_src}
                p.fuStride  = make_longlong3(fustride1, fustride0, 0);
                p.fdStride  = make_longlong3(fdstride1, fdstride0, 0);

                // Determine if indices don't fit in int32. Support negative strides although Torch currently never produces those.
                bool index64b = false;
                // if (std::abs(p.bStride * x_shape1) > INT_MAX) index64b = true;
                // if (std::min(x_shape0 * p.xStride.w, 0ll) + std::min(x_shape1 * p.xStride.z, 0ll) + std::min(x_shape2 * p.xStride.y, 0ll) + std::min(x_shape3 * p.xStride.x, 0ll) < -INT_MAX) index64b = true;
                // if (std::max(x_shape0 * p.xStride.w, 0ll) + std::max(x_shape1 * p.xStride.z, 0ll) + std::max(x_shape2 * p.xStride.y, 0ll) + std::max(x_shape3 * p.xStride.x, 0ll) >  INT_MAX) index64b = true;
                // if (std::min(y_shape0 * p.yStride.w, 0ll) + std::min(y_shape1 * p.yStride.z, 0ll) + std::min(y_shape2 * p.yStride.y, 0ll) + std::min(y_shape3 * p.yStride.x, 0ll) < -INT_MAX) index64b = true;
                // if (std::max(y_shape0 * p.yStride.w, 0ll) + std::max(y_shape1 * p.yStride.z, 0ll) + std::max(y_shape2 * p.yStride.y, 0ll) + std::max(y_shape3 * p.yStride.x, 0ll) >  INT_MAX) index64b = true;
                // if (s.numel() > INT_MAX) index64b = true;

                // Choose CUDA kernel.
                filtered_lrelu_kernel_spec spec = {{ 0 }};
                if      (!index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, true,  false>(p, sharedKB);
                else if (!index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, false, true >(p, sharedKB);
                else if (!index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
                // assert(spec.exec, "internal error - CUDA kernel not found") // This should not happen because we tested earlier that kernel exists.

                // Launch CUDA kernel.
                void* args[] = {{&p}};
                int bx = spec.numWarps * 32;
                int gx = (p.yShape.x - 1) / spec.tileOut.x + 1;
                int gy = (p.yShape.y - 1) / spec.tileOut.y + 1;
                int gz = p.yShape.z * p.yShape.w;

                // Repeat multiple horizontal tiles in a CTA?
                if (spec.xrep)
                {{
                    p.tilesXrep = spec.xrep;
                    p.tilesXdim = gx;

                    gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
                    std::swap(gx, gy);
                }}
                else
                {{
                    p.tilesXrep = 0;
                    p.tilesXdim = 0;
                }}

                // Launch filter setup kernel.
                CUDA_CHECK(cudaLaunchKernel(spec.setup, 1, 1024, args, 0, stream));

                // Copy kernels to constant memory.
                if ( writeSigns && !readSigns) {{
                    CUDA_CHECK((copy_filters<true,  false>(stream)));
                }}
                else if (!writeSigns && readSigns) {{
                    CUDA_CHECK((copy_filters<false, true >(stream)));
                }}
                else if (!writeSigns && !readSigns) {{
                    CUDA_CHECK((copy_filters<false, false>(stream)));
                }}

                // Set cache and shared memory configurations for main kernel.
                CUDA_CHECK(cudaFuncSetCacheConfig(spec.exec, cudaFuncCachePreferShared));
                if (spec.dynamicSharedKB) // Need dynamically allocated shared memory?
                CUDA_CHECK(cudaFuncSetAttribute(spec.exec, cudaFuncAttributeMaxDynamicSharedMemorySize, spec.dynamicSharedKB << 10));
                CUDA_CHECK(cudaFuncSetSharedMemConfig(spec.exec, cudaSharedMemBankSizeFourByte));

                // Launch main kernel.
                const int maxSubGz = 65535; // CUDA maximum for block z dimension.
                for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
                {{
                    p.blockZofs = zofs;
                    int subGz = std::min(maxSubGz, gz - zofs);
                    CUDA_CHECK(cudaLaunchKernel(spec.exec, dim3(gx, gy, subGz), bx, args, spec.dynamicSharedKB << 10, stream));
                }}
            }}
        '''

        fut_w = self.fu.shape[-1] - 1
        fut_h = self.fu.shape[0] - 1
        fdt_w = self.fd.shape[-1] - 1
        fdt_h = self.fd.shape[0] - 1 
        cw = x.shape[3] * self.up + (self.px0 + self.px1) - fut_w
        ch = x.shape[2] * self.up + (self.py0 + self.py1) - fut_h
        yw = (cw - fdt_w + (self.down - 1)) // self.down
        yh = (ch - fdt_h + (self.down - 1)) // self.down
        self.y_shape = (x.shape[0], x.shape[1], yh, yw)
        sw_active = yw * self.down - (self.down - 1) + fdt_w
        sh = yh * self.down - (self.down - 1) + fdt_h
        sw = (sw_active + 15) & ~15
        s_shape = (x.shape[0], x.shape[1], sh, sw // 4)
        output, sign_output = jt.code([self.y_shape, s_shape],[x.dtype, jt.uint8],[x, self.fu, self.fd, bias, self.si],cuda_header=cuda_header,cuda_src=cuda_src)
        # ret.sync()
        if output.numel() == 0: # no valid kernel!!!
            self.ret = 1
            x = x + bias.unsqueeze(-1).unsqueeze(-1)
            ups = Upfirdn2d(up=self.up, padding=[self.px0, self.px1, self.py0, self.py1], gain=self.up**2, flip_filter=self.flip_filter)
            ups2 = Upfirdn2d(down=self.down, flip_filter=self.flip_filter)
            with jt.no_grad():
                y = ups(x, self.fu)
                # print(y)
                y, sign_output = filter_lrelu_act(y, self.si, self.sx, self.sy, self.gain, self.slope, self.clamp, writeSigns, self.y_shape, s_shape)
                # print(y)
                output = ups2(y, self.fd)
        else:
            self.ret = 0
        self.x_shape = x.shape
        self.x_dtype = x.dtype
        # print(sign_output)
        self.save_so = sign_output
        return output

    def grad(self, grads):  # only support backward for bias and input now (in pytorch.)
        #print(grads)
        #print("jittor: ", grads.shape)
        _, _, xh, xw = self.x_shape
        _, _, yh, yw = self.y_shape
        pp = [
            (self.fu.shape[-1] - 1) + (self.fd.shape[-1] - 1) - self.px0,
            xw * self.up - yw * self.down + self.px0 - (self.up - 1),
            (self.fu.shape[0] - 1) + (self.fd.shape[0] - 1) - self.py0,
            xh * self.up - yh * self.down + self.py0 - (self.up - 1),
        ]
        gg = self.gain * (self.up ** 2) / (self.down ** 2)
        ff = 1 if not self.flip_filter else 0
        sx = self.sx - (self.fu.shape[-1] - 1) + self.px0
        sy = self.sy - (self.fu.shape[0]  - 1) + self.py0
        #print("jittor: ", grads.shape)
        # print("jittor: ", pp, gg, ff, sx, sy)
        if self.ret == 1:
            # no valid kernel in forward. Use upfirdn2d instead.
            ups = Upfirdn2d(up=self.down, padding=[pp[0], pp[1], pp[2], pp[3]], gain=self.down**2, flip_filter=ff)
            ups2 = Upfirdn2d(down=self.up, flip_filter=ff)
            with jt.no_grad():
                y = ups(grads, self.fd)
                y, sign_output = filter_lrelu_act(y, self.save_so, sx, sy, gg, self.slope, self.clamp, 0, y.shape, (1,1,1,1))
                dx = ups2(y, self.fu)
            db = dx.sum([0, 2, 3])
            return dx, db
        cuda_header = code_op_read_file()[0]
        if self.fd.ndim == 2:
            fushape_src = '''
            int fushape1 = fu_inp_shape1;
            int fushape0 = fu_inp_shape0;
            '''
            fustride_src = '''
            int fustride1 = fu_inp_stride1;
            int fustride0 = fu_inp_stride0;
            '''
        else:
            fushape_src = '''
            int fushape1 = fu_inp_shape0;
            int fushape0 = 0;
            '''
            fustride_src = '''
            int fustride1 = fu_inp_stride0;
            int fustride0 = 0;
            '''
        if self.fu.ndim == 2:
            fdshape_src = '''
            int fdshape1 = fd_inp_shape1;
            int fdshape0 = fd_inp_shape0;
            '''
            fdstride_src = '''
            int fdstride1 = fd_inp_stride1;
            int fdstride0 = fd_inp_stride0;
            '''
        else:
            fdshape_src = '''
            int fdshape1 = fd_inp_shape0;
            int fdshape0 = 0;
            '''
            fdstride_src = '''
            int fdstride1 = fd_inp_stride0;
            int fdstride0 = 0;
            '''
        cuda_src = f'''
            @alias(x_inp, in0)
            @alias(fu_inp, in1)
            @alias(fd_inp, in2)
            @alias(b_inp, in3)
            @alias(s_inp, in4)
            @alias(y_oup, out0)
            cudaStream_t stream = 0;
            int up = {self.down};
            int down = {self.up};
            int px0,py0,px1,py1;
            px0 = {pp[0]};
            px1 = {pp[1]};
            py0 = {pp[2]};
            py1 = {pp[3]};
            float gain = {gg};
            float slope = {self.slope};
            float clamp = {self.clamp};
            bool flip_filter = {ff};
            bool writeSigns = {0};
            int sx = {sx};
            int sy = {sy};
            // Figure out how much shared memory is available on the device.
            int maxSharedBytes = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
            int sharedKB = maxSharedBytes >> 10;
            // Populate enough launch parameters to check if a CUDA kernel exists.
            filtered_lrelu_kernel_params p;
            p.up      = up;
            p.down    = down;
            {fushape_src}
            {fdshape_src}
            p.fuShape = make_int2(fushape1, fushape0); // shape [n, 0] indicates separable filter.
            p.fdShape = make_int2(fdshape1, fdshape0);
            filtered_lrelu_kernel_spec test_spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
            if (!test_spec.exec)
            {{
                // No kernel found - return empty tensors and indicate missing kernel with return code of -1.
                LOGir << "Backward No kernel found! Back to generic mode!!";
            }} else {{
                // Input/output element size.
                // int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;
                int64_t sz = 4; // support float32 for now.
                // Input sizes.
                int64_t xw = (int)x_inp_shape3;
                int64_t xh = (int)x_inp_shape2;
                int64_t fut_w = fushape1 - 1;
                int64_t fut_h = (int)fu_inp_shape0  - 1;
                int64_t fdt_w = fdshape1 - 1;
                int64_t fdt_h = (int)fd_inp_shape0  - 1;

                // Logical size of upsampled buffer.
                int64_t cw = xw * up + (px0 + px1) - fut_w;
                int64_t ch = xh * up + (py0 + py1) - fut_h;
                // assert(cw > fdt_w && ch > fdt_h, "upsampled buffer must be at least the size of downsampling filter");
                // assert(cw <= INT_MAX && ch <= INT_MAX, "upsampled buffer is too large");

                // Compute output size and allocate.
                int64_t yw = y_oup_shape3;
                int64_t yh = y_oup_shape2;
                // assert(yw > 0 && yh > 0, "output must be at least 1x1");
                // assert(yw <= INT_MAX && yh <= INT_MAX, "output is too large");
                // LOGir << "jittor bk shape" << fut_w << " " << fut_h << " " << fdt_w << " " << fdt_h << " " << cw << " " << ch;
                // Allocate sign tensor.
                bool readSigns = !!{self.save_so.numel()};
                int64_t sw_active = 0; // Active width of sign tensor.
                uint8_t* so = s_inp_p;
                int s_shape3, s_shape2;
                if(!!{self.save_so.numel()}) {{
                    s_shape3 = s_inp_shape3;
                    s_shape2 = s_inp_shape2;
                }}
                // LOGir << "jtgrad " << s_shape3 << " " << s_shape2;
                sw_active = s_inp_shape3 << 2;
                // Populate rest of CUDA kernel parameters.
                p.x         = x_inp_p;
                p.y         = y_oup_p;
                p.b         = b_inp_p;
                p.s         = (readSigns || writeSigns) ? (unsigned char*)so : 0;
                p.fu        = (float*) fu_inp_p;
                p.fd        = (float*) fd_inp_p;
                p.pad0      = make_int2(px0, py0);
                p.gain      = gain;
                p.slope     = slope;
                p.clamp     = clamp;
                p.flip      = (flip_filter) ? 1 : 0;
                p.xShape    = make_int4((int)x_inp_shape3, (int)x_inp_shape2, (int)x_inp_shape1, (int)x_inp_shape0);
                p.yShape    = make_int4((int)y_oup_shape3, (int)y_oup_shape2, (int)y_oup_shape1, (int)y_oup_shape0);
                p.sShape    = (readSigns || writeSigns) ? make_int2((int)s_shape3, (int)s_shape2) : make_int2(0, 0); // Width is in bytes. Contiguous.
                p.sOfs      = make_int2(sx, sy);
                p.swLimit   = (sw_active + 3) >> 2; // Rounded up to bytes.
                // x, y, b strides are in bytes.
                p.xStride   = make_longlong4(sz * x_inp_stride3, sz * x_inp_stride2, sz * x_inp_stride1, sz * x_inp_stride0);
                p.yStride   = make_longlong4(sz * y_oup_stride3, sz * y_oup_stride2, sz * y_oup_stride1, sz * y_oup_stride0);
                p.bStride   = sz * b_inp_stride0;

                // fu, fd strides are in elements.
                {fustride_src}
                {fdstride_src}
                p.fuStride  = make_longlong3(fustride1, fustride0, 0);
                p.fdStride  = make_longlong3(fdstride1, fdstride0, 0);
                // Determine if indices don't fit in int32. Support negative strides although Torch currently never produces those.
                bool index64b = false;
                // if (std::abs(p.bStride * x_shape1) > INT_MAX) index64b = true;
                // if (std::min(x_shape0 * p.xStride.w, 0ll) + std::min(x_shape1 * p.xStride.z, 0ll) + std::min(x_shape2 * p.xStride.y, 0ll) + std::min(x_shape3 * p.xStride.x, 0ll) < -INT_MAX) index64b = true;
                // if (std::max(x_shape0 * p.xStride.w, 0ll) + std::max(x_shape1 * p.xStride.z, 0ll) + std::max(x_shape2 * p.xStride.y, 0ll) + std::max(x_shape3 * p.xStride.x, 0ll) >  INT_MAX) index64b = true;
                // if (std::min(y_shape0 * p.yStride.w, 0ll) + std::min(y_shape1 * p.yStride.z, 0ll) + std::min(y_shape2 * p.yStride.y, 0ll) + std::min(y_shape3 * p.yStride.x, 0ll) < -INT_MAX) index64b = true;
                // if (std::max(y_shape0 * p.yStride.w, 0ll) + std::max(y_shape1 * p.yStride.z, 0ll) + std::max(y_shape2 * p.yStride.y, 0ll) + std::max(y_shape3 * p.yStride.x, 0ll) >  INT_MAX) index64b = true;
                // if (s.numel() > INT_MAX) index64b = true;

                // Choose CUDA kernel.
                filtered_lrelu_kernel_spec spec = {{ 0 }};
                if      (!index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, true,  false>(p, sharedKB);
                else if (!index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, false, true >(p, sharedKB);
                else if (!index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
                assert(spec.exec); // This should not happen because we tested earlier that kernel exists.
                // Launch CUDA kernel.
                void* args[] = {{&p}};
                int bx = spec.numWarps * 32;
                int gx = (p.yShape.x - 1) / spec.tileOut.x + 1;
                int gy = (p.yShape.y - 1) / spec.tileOut.y + 1;
                int gz = p.yShape.z * p.yShape.w;

                // Repeat multiple horizontal tiles in a CTA?
                if (spec.xrep)
                {{
                    p.tilesXrep = spec.xrep;
                    p.tilesXdim = gx;

                    gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
                    std::swap(gx, gy);
                }}
                else
                {{
                    p.tilesXrep = 0;
                    p.tilesXdim = 0;
                }}

                // Launch filter setup kernel.
                CUDA_CHECK(cudaLaunchKernel(spec.setup, 1, 1024, args, 0, stream));

                // Copy kernels to constant memory.
                if ( writeSigns && !readSigns) {{
                    CUDA_CHECK((copy_filters<true,  false>(stream)));
                }}
                else if (!writeSigns && readSigns) {{
                    CUDA_CHECK((copy_filters<false, true >(stream)));
                }}
                else if (!writeSigns && !readSigns) {{
                    CUDA_CHECK((copy_filters<false, false>(stream)));
                }}

                // Set cache and shared memory configurations for main kernel.
                CUDA_CHECK(cudaFuncSetCacheConfig(spec.exec, cudaFuncCachePreferShared));
                if (spec.dynamicSharedKB) // Need dynamically allocated shared memory?
                CUDA_CHECK(cudaFuncSetAttribute(spec.exec, cudaFuncAttributeMaxDynamicSharedMemorySize, spec.dynamicSharedKB << 10));
                CUDA_CHECK(cudaFuncSetSharedMemConfig(spec.exec, cudaSharedMemBankSizeFourByte));

                // Launch main kernel.
                const int maxSubGz = 65535; // CUDA maximum for block z dimension.
                for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
                {{  
                    p.blockZofs = zofs;
                    int subGz = std::min(maxSubGz, gz - zofs);
                    CUDA_CHECK(cudaLaunchKernel(spec.exec, dim3(gx, gy, subGz), bx, args, spec.dynamicSharedKB << 10, stream));
                }}
            }}
        '''
        
        '''
        print("jittor before jt.code")
        memory = {}
        memory['x_shape'] = self.x_shape
        memory['x_dtype'] = self.x_dtype
        #print("time sleep")
        memory['grads'] = grads
        memory['fd'] = self.fd
        memory['fu'] = self.fu
        print("before zero")
        memory['zeros'] = jt.zeros(self.x_shape[1])
        print("end zero")
        memory['save_so'] = self.save_so
        memory['cuda_header'] = cuda_header
        memory['cuda_src'] = cuda_src
        jt.save(memory, './home/liufenglin/210_16T/code/jittor_repositories/DeepFaceVideoEditingDebug_no_weight/memory.pkl')
        time.sleep(20000)
        print("time sleep")
        '''
        
        #test =  jt.zeros(self.x_shape[1])
        dx = jt.code(self.x_shape, self.x_dtype, [grads, self.fd, self.fu, jt.zeros(self.x_shape[1]), self.save_so], cuda_header=cuda_header, cuda_src=cuda_src)
        #print("jittor after jt.code")
        db = dx.sum([0, 2, 3])
        #print("jittor over: ", dx.shape)
        return dx, db


if __name__ == "__main__":
    # x = jt.array(np.random.uniform(0,1,(1,1,3,3)))
    nx = np.random.uniform(0,0.1,(1,3,1024,1024,)).astype("float32")
    nfu = np.random.uniform(-0.1,0.1,(12,)).astype("float32")
    nfd = np.random.uniform(-0.1,0.1,(12,)).astype("float32")
    nbias = np.random.uniform(-0.1,0.1,(3,)).astype("float32")
    # # nbias = np.zeros((512,)).astype("float32")
    x = jt.array(nx)
    fu = jt.array(nfu)
    fd = jt.array(nfd)
    # fu = None
    # fd = None
    bias = jt.array(nbias)
    up = 2
    down = 2
    pad = [-6, -9, -6, -9]
    frelu = Filtered_LReLU(fu=fu, fd=fd, up=up, down=down, padding=pad)
    out = frelu(x, bias)
    # # print(out)
    grad = jt.grad(out, [x, bias])
    print(out, grad)
    jt.sync_all()
    # # print(grad)
    # x = Variable((torch.from_numpy(nx)), requires_grad=True).cuda()
    # # fu = torch.from_numpy(nfu).cuda()
    # # fd = torch.from_numpy(nfd).cuda()
    # fu = None
    # fd = None
    # bias = Variable(torch.from_numpy(nbias), requires_grad=True).cuda()
    # output = filtered_lrelu.filtered_lrelu(x, fu=fu, fd=fd, b=bias, up=up, down=down, padding=pad, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda')
    # # print(output)
    # tgq = torch.autograd.grad(output, x, torch.ones_like(output), retain_graph=True)
    # tgr = torch.autograd.grad(output, bias, torch.ones_like(output))
    # # print(tgq)
    # np.testing.assert_allclose(out.numpy(), output.detach().cpu().numpy())
    # print("pass forward")
    # np.testing.assert_allclose(grad[0].numpy(),tgq[0].detach().cpu().numpy(), rtol=1e-3, atol=2e-3)
    # print("pass input backward")
    # np.testing.assert_allclose(grad[1].numpy(),tgr[0].detach().cpu().numpy(), rtol=1e-3, atol=2e-3)
    # print("pass bias backward")