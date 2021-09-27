import torch
import math
irange = range
import os
import random
import shutil
import json
import progressbar
import torchvision
#import fastBPE

#import TransCoder.preprocessing.src.code_tokenizer as code_tokenizer
#from TransCoder.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GradSaver:
    def __init__(self):
        self.grad = -1
    
    def save_grad(self, grad):
        self.grad = grad

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    #print('bs:', batch_size)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    #print('ind: ', ind.shape)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    #print('correct: ', correct.shape)
    #print(correct)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

# class PreprocessorCPP(object):
#     def __init__(self, model_path, BPE_path):
#         reloaded = torch.load(model_path, map_location='cpu')
#         reloaded_params = AttrDict(reloaded['params'])
#         self.dico = Dictionary(
#                 reloaded['dico_id2word'],
#                 reloaded['dico_word2id'],
#                 reloaded['dico_counts']
#                 )

#         self.bpe_model = fastBPE.fastBPE(os.path.abspath(BPE_path))
        
#         lang = 'cpp'
#         self.tokenizer = getattr(code_tokenizer, f'tokenize_{lang}')
#         self.detokenizer = getattr(code_tokenizer, f'detokenize_{lang}')

#         lang += '_sa'
#         lang_id = reloaded_params.lang2id[lang]

#     def word_to_index(self, w):
#         return self.dico.index(w)

#     def index_to_word(self, idx):
#         return self.dico[idx]

#     def preprocess(self, input_code):
#         tokens = [t for t in self.tokenizer(input_code)]
#         tokens = self.bpe_model.apply(tokens)
#         tokens = ['</s>'] + tokens + ['</s>']
#         out = [self.dico.index(w) for w in tokens]
#         return out
#         # out = torch.LongTensor([self.dico.index(w)
#         #                         for w in tokens])[:, None]

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)

def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        #return super().__call__(tensor)

def my_scale(v, v_max, v_min, low=0, up=1):
    return (up - low) * (v - v_min) / max(1e-7, v_max - v_min) + low

def my_scale_inv(v, v_max, v_min, low=0, up=1):
    return (v - low) / (up - low) * max(1e-7, v_max - v_min) + v_min

def get_widgets():
    return ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def reparameterize(mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = logvar.data.new(logvar.size()).normal_()
    return eps.mul(logvar).add_(mu)

def KLDLoss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0).cuda()

# def accuracy(preds, y):
#     preds = torch.argmax(preds, dim=1)
#     correct = (preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc

def build_vocab(vocab_path, token_list):
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            token_vocab = vocab['token']
            index_vocab = vocab['index']
    else:
        token_vocab = {}
        index_vocab = {}
        count = {}
        cur_index = 0
        for token in token_list:
            for t in token:
                if t in count.keys():
                    count[t] += 1
                    if count[t] == 10:
                        token_vocab[t] = cur_index
                        index_vocab[cur_index] = t
                        cur_index += 1
                else:
                    count[t] = 1
        for t in ['<UNK>', '<START>', '<END>']:
            token_vocab[t] = cur_index
            index_vocab[cur_index] = t
            cur_index += 1
        with open(vocab_path, 'w') as f:
            json.dump({'token': token_vocab,
                       'index': index_vocab}, f)
    return token_vocab, index_vocab

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)