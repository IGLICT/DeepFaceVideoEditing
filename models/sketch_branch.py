import jittor as jt
import jittor.nn as nn
import numpy as np

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        conv_block += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        ]

        conv_block = nn.Sequential(*conv_block)

        return conv_block

    def execute(self, x):
        out = x + self.conv_block(x)
        return out

class SketchBlock(nn.Module):
    def __init__(self, prev_channel=512, inter_channel=256, sketch_channel=32):
        super().__init__()
        norm_layer=nn.BatchNorm2d

        self.out_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_channel, inter_channel, kernel_size=3, padding=0),
            nn.InstanceNorm2d(inter_channel),
        )

        self.out_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inter_channel, sketch_channel, kernel_size=3, padding=0),
            nn.InstanceNorm2d(sketch_channel),
        )
    
    def execute(self, x):
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        return x

class SketchGenerator(nn.Module):
    def __init__(self, img_resolution = 1024, sketch_channel=32, output_nc = 1):
        super().__init__()
        dims =       [512,512,512,512,512,512,512,512,323,203,128,81,51,32,32]
        inter_dims = [256,256,256,256,256,256,256,256,256,128, 64,64,32,32,32]
        res = [36,36,36,52,52,84,148,148,276,276,532,1044,1044,1044]

        for i in range(len(dims)):
            block = SketchBlock(dims[i], inter_dims[i], sketch_channel)
            setattr(self, f'b{i}', block)
        
        to_sketch = [nn.ReflectionPad2d(3), nn.Conv2d(sketch_channel, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.to_sketch = nn.Sequential(*to_sketch)

    def execute(self, feature_list):
        block = getattr(self, f'b{0}')
        sketch_fea = block(feature_list[0][:,:,10:10 + 16,10:10 + 16])
        block = getattr(self, f'b{1}')
        sketch_fea += block(feature_list[1][:,:,10:10 + 16,10:10 + 16])
        block = getattr(self, f'b{2}')
        sketch_fea += block(feature_list[2][:,:,10:10 + 16,10:10 + 16])
        
        sketch_fea = nn.interpolate(sketch_fea, [32, 32], mode='nearest')
        block = getattr(self, f'b{3}')
        sketch_fea += block(feature_list[3][:,:,10:10 + 32,10:10 + 32])
        block = getattr(self, f'b{4}')
        sketch_fea += block(feature_list[4][:,:,10:10 + 32,10:10 + 32])

        sketch_fea = nn.interpolate(sketch_fea, [64, 64], mode='nearest')
        block = getattr(self, f'b{5}')
        sketch_fea += block(feature_list[5][:,:,10:10 + 64,10:10 + 64])

        sketch_fea = nn.interpolate(sketch_fea, [128, 128], mode='nearest')
        block = getattr(self, f'b{6}')
        sketch_fea += block(feature_list[6][:,:,10:10 + 128,10:10 + 128])
        block = getattr(self, f'b{7}')
        sketch_fea += block(feature_list[7][:,:,10:10 + 128,10:10 + 128])

        sketch_fea = nn.interpolate(sketch_fea, [256, 256], mode='nearest')
        block = getattr(self, f'b{8}')
        sketch_fea += block(feature_list[8][:,:,10:10 + 256,10:10 + 256])
        block = getattr(self, f'b{9}')
        sketch_fea += block(feature_list[9][:,:,10:10 + 256,10:10 + 256])
        
        sketch_fea = nn.interpolate(sketch_fea, [512, 512], mode='nearest')
        block = getattr(self, f'b{10}')
        sketch_fea += block(feature_list[10][:,:,10:10 + 512,10:10 + 512])

        block = getattr(self, f'b{11}')
        feature = nn.interpolate(feature_list[11][:,:,10:10 + 1024,10:10 + 1024], [512, 512], mode='nearest')
        sketch_fea += block(feature)

        block = getattr(self, f'b{12}')
        feature = nn.interpolate(feature_list[12][:,:,10:10 + 1024,10:10 + 1024], [512, 512], mode='nearest')
        sketch_fea += block(feature)

        block = getattr(self, f'b{13}')
        feature = nn.interpolate(feature_list[13][:,:,10:10 + 1024,10:10 + 1024], [512, 512], mode='nearest')
        sketch_fea += block(feature)

        output_sketch = self.to_sketch(sketch_fea)
        return output_sketch

