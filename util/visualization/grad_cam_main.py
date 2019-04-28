# Chinko Group
# based on
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os
import cv2
import numpy as np
import torch
import argparse
from torch.autograd import Variable
from torchvision import models, transforms, datasets
from os.path import isfile, join
from os import listdir

from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)
#from grad_cam import (resnet50, resnet152)

script_dir = os.path.dirname(__file__)

def to_var(image):
    return Variable(image.unsqueeze(0), volatile=False, requires_grad=True)


def save_gradient(filename, data, output_path):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    path = os.path.join(output_path, filename)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite((path), np.uint8(data))


def save_gradcam(filename, gcam, raw_image, output_path):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    if(gcam.max() != 0):
        gcam = gcam / gcam.max() * 255.0

    path = os.path.join(output_path, filename)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, np.uint8(gcam))


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--images_path', default='/grad_cam_testset/', type=str, help='imagefolder path')
parser.add_argument('--output_path', default='/', type=str, help='imagefolder path')
parser.add_argument('--model_path', default='/model/model_best.pth.tar', type=str, help='model path')
parser.add_argument('--arch', default='resnet152', type=str, help='resnet model arch')
parser.add_argument('--gpu_id', default='3', type=str, help='1,2')
parser.add_argument('--test', default='3', type=str, help='1,2')
parser.add_argument('--topk', default=1, type=int, help='3')
parser.add_argument('--n_classes', default=8, type=int, help='3')
parser.add_argument('--layer', default='module.layer4.2', type=str, help='3')
args = parser.parse_args()


def main():
    image_path = os.path.join(script_dir, args.images_path)
    model_path = os.path.join(script_dir, args.model_path)
    output_path = os.path.join(script_dir, args.output_path)
    arch = args.arch
    topk = args.topk
    n_classes = args.n_classes
    cuda = True
    test = args.test
    layer = args.layer
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    images = [join(dp, f) for dp, dn, filenames in os.walk(image_path) for f in filenames]

    cuda = cuda and torch.cuda.is_available()

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    classes = range(0, n_classes)
    try:
        H = datasets.ImageFolder(image_path)
        classes = H.classes
    except:
        print('could not load classes, used default ones')

    print(classes)

    model = models.__dict__[arch](num_classes=n_classes, pretrained=False)
    model = torch.nn.DataParallel(model)

    #model_dict = torch.load('/home/chinko/gradcam2/checkpoint/model_best.pth.tar')
    #model_dict = torch.load(model_path)
    #model = resnet50(False, output_channels=8)

    print('Loading a saved model')
    print(model_path)

    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove 'module.' of dataparallel
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict, strict=True)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if cuda:
        model = model.cuda()

    for img_path in images:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        # Image
        raw_image = cv2.imread(img_path)[..., ::-1]
        raw_image = cv2.resize(raw_image, (224, ) * 2)
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.468, 0.451, 0.388],
                std=[0.298, 0.300, 0.311],
            )
        ])(raw_image)

        if cuda:
            image = image.cuda()

        # =========================================================================
        print('Grad-CAM')
        # =========================================================================
        gcam = GradCAM(model=model)
        probs, idx = gcam.forward(to_var(image))

        for i in range(0, topk):
            gcam.backward(idx=idx[i])
            output = gcam.generate(target_layer=layer)
            prob = '[{:.5f}]'.format(probs[i])
            save_gradcam('results_{}/{}_{}_gcam_{}_{}.png'.format(test, classes[idx[i]], prob, arch, filename), output, raw_image, output_path)
            print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

        # =========================================================================
        print('Guided Backpropagation/Guided Grad-CAM')
        # =========================================================================
        gbp = GuidedBackPropagation(model=model)
        probs, idx = gbp.forward(to_var(image))

        for i in range(0, topk):
            gcam.backward(idx=idx[i])
            region = gcam.generate(target_layer=layer)
            gbp.backward(idx=idx[i])
            feature = gbp.generate()
            h, w, _ = feature.shape
            region = cv2.resize(region, (w, h))[..., np.newaxis]
            output = feature * region
            prob = '[{:.5f}]'.format(probs[i])
            save_gradient('results_{}/{}_{}_ggcam_{}_{}.png'.format(test, classes[idx[i]], prob, arch, filename), output, output_path)

        gcam.clear()
        gbp.clear()

if __name__ == '__main__':
    main()