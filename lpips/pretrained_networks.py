import jittor as jt
from jittor import models
from collections import namedtuple
#import jittor

#Alexnet = jt.models.alexnet(pretrained=True)
#print(Alexnet)

class squeezenet(jt.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = jt.models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = jt.nn.Sequential()
        self.slice2 = jt.nn.Sequential()
        self.slice3 = jt.nn.Sequential()
        self.slice4 = jt.nn.Sequential()
        self.slice5 = jt.nn.Sequential()
        self.slice6 = jt.nn.Sequential()
        self.slice7 = jt.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out

class alexnet(jt.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = jt.models.alexnet(pretrained=pretrained).features
        self.slice1 = jt.nn.Sequential()
        self.slice2 = jt.nn.Sequential()
        self.slice3 = jt.nn.Sequential()
        self.slice4 = jt.nn.Sequential()
        self.slice5 = jt.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vgg16(jt.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = jt.models.vgg16(pretrained=pretrained).features
        self.slice1 = jt.nn.Sequential()
        self.slice2 = jt.nn.Sequential()
        self.slice3 = jt.nn.Sequential()
        self.slice4 = jt.nn.Sequential()
        self.slice5 = jt.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

if __name__ == "__main__":
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
    ex_ref = read_img('./imgs/ex_ref.png')
    #print(ex_ref)

    net_type = 'squeeze'
    if net_type == 'alex':
        net = alexnet()
        out = net(ex_ref)
        print(out)
    elif net_type == 'vgg':
        net = vgg16()
        out = net(ex_ref)
        print(out)
    elif net_type == 'squeeze':
        net = squeezenet()
        out = net(ex_ref)
        print(out)
    else:
        print("no implementation")



