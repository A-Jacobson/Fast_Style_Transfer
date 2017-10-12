from torch.autograd import Variable
import numpy as np
import torch
from PIL import Image

def imagenet_preprocess(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    std = tensortype(batch.data.size())

    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406

    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    return (batch - Variable(mean)) / Variable(std)

def to_pil_image(pic):
    """Convert a tensor or an ndarray to PIL Image.
    See ``ToPIlImage`` for more details.
    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.
    Returns:
        PIL.Image: Image converted to PIL.Image.
    """

    pic = pic.mul(255).byte()
    npimg = np.transpose(pic.numpy(), (1, 2, 0)).clip(0, 255)

    assert isinstance(npimg, np.ndarray)
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]

        if npimg.dtype == np.uint8:
            mode = 'L'
        if npimg.dtype == np.int16:
            mode = 'I;16'
        if npimg.dtype == np.int32:
            mode = 'I'
        elif npimg.dtype == np.float32:
            mode = 'F'
    elif npimg.shape[2] == 4:
            if npimg.dtype == np.uint8:
                mode = 'RGBA'
    else:
        if npimg.dtype == np.uint8:
            mode = 'RGB'
    assert mode is not None, '{} is not supported'.format(npimg.dtype)
    return Image.fromarray(npimg, mode=mode)


def gram_matrix(img):
    (b, ch, h, w) = img.size()
    features = img.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features @ features_t / (ch * h * w)
