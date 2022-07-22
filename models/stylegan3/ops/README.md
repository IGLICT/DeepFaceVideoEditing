# jittor-filtered-lrelu

Jittor's implementation of filtered_lrelu ops. Original implementation is in this link: https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/filtered_lrelu.cu. 

Only support CUDA and float32 for now. fd, fu's ndim should be 2. support generic input and output shape for now.