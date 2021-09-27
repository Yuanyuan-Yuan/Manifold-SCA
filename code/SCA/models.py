import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import copy

from cbam import CBAM
from bam import BAM

class Attention(nn.Module):
    """
    Attention Network.
    Codes from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    Codes from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class FCs(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=100):
        super(FCs, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(in_ch, h_ch),
                nn.ReLU(),
                nn.Linear(h_ch, out_ch),
                nn.ReLU()
            )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        m.bias.data.fill_(0)

def weights_init_rnn(model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class image_decoder_32(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(image_decoder_32, self).__init__()
        self.dim = dim
        self.nc = nc
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Sigmoid()
        )
        self.main = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        output = self.fc(output.view(-1, 1024))
        return output

class image_decoder_128(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(image_decoder_128, self).__init__()
        self.dim = dim
        self.nc = nc
        self.main = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class attn_trace_encoder_512(nn.Module):
    def __init__(self, dim, nc=1):
        super(attn_trace_encoder_512, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 512 x 512
        self.c1 = dcgan_conv(nc, nf)
        # input is (nf) x 256 x 256
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 128 x 128
        self.c3 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c4 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c5 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c6 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c7 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c8 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf)
        self.attn3 = CBAM(nf)
        self.attn4 = CBAM(nf * 2)
        self.attn5 = CBAM(nf * 4)
        self.attn6 = CBAM(nf * 8)
        self.attn7 = CBAM(nf * 8)
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.attn7(self.c7(h6))
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x = F.normalize(x)
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.attn7(self.c7(h6))
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

class trace_encoder_512(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_512, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 512 x 512
        self.c1 = dcgan_conv(nc, nf)
        # input is (nf) x 256 x 256
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 128 x 128
        self.c3 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c4 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c5 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c6 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c7 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c8 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

class attn_trace_encoder_256(nn.Module):
    def __init__(self, dim, nc=1):
        super(attn_trace_encoder_256, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 256 x 256
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 128 x 128
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c4 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c5 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c7 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf)
        self.attn3 = CBAM(nf * 2)
        self.attn4 = CBAM(nf * 4)
        self.attn5 = CBAM(nf * 8)
        self.attn6 = CBAM(nf * 8)
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.c7(h6)
        return h7.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x = F.normalize(x)
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.c7(h6)
        return h7.view(-1, self.dim)

class trace_encoder_256(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_256, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 256 x 256
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 128 x 128
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c4 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c5 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c7 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)
        return h7.view(-1, self.dim)

class trace_encoder_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class attn_trace_encoder_square_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(attn_trace_encoder_square_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf * 2)
        self.attn3 = CBAM(nf * 4)
        self.attn4 = CBAM(nf * 8)
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)#.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)#.view(-1, self.dim)

class trace_encoder_square_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_square_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        # self.c5 = dcgan_conv(nf * 8, nf * 8)
        # # state size. (nf*8) x 4 x 4
        # self.c6 = nn.Sequential(
        #         nn.Conv2d(nf * 8, dim, 4, 1, 0),
        #         nn.BatchNorm2d(dim),
        #         nn.Tanh()
        #         )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)#.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x = F.normalize(x)
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)#.view(-1, self.dim)

class audio_decoder_128(nn.Module):
    def __init__(self, dim, nc=1, out_s=44):
        super(audio_decoder_128, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, (4, 4), (1, 2), (0, 1)),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 2
        self.upc2 = dcgan_upconv(nf * 8, nf * 8)
        # state size. (nf*8) x 8 x 4
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 16 x 8
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 32 x 16
        self.upc5 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 64 x 32
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.BatchNorm2d(nc),
                nn.ReLU()
                #nn.Tanh()
                #nn.Sigmoid()
                # state size. (nc) x 128 x 64
                )
        self.fc = nn.Sequential(
                nn.Linear(64, out_s),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        d6 = self.upc6(d5)
        out = self.fc(d6)
        return out

# class image_decoder_128(nn.Module):
#     def __init__(self, dim, nc=1):
#         super(image_decoder_128, self).__init__()
#         self.dim = dim
#         nf = 64
#         self.upc1 = nn.Sequential(
#                 # input is Z, going into a convolution
#                 nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
#                 nn.BatchNorm2d(nf * 8),
#                 nn.ReLU()
#                 )
#         # state size. (nf*8) x 4 x 2
#         self.upc2 = dcgan_upconv(nf * 8, nf * 8)
#         # state size. (nf*8) x 8 x 4
#         self.upc3 = dcgan_upconv(nf * 8, nf * 4)
#         # state size. (nf*4) x 16 x 8
#         self.upc4 = dcgan_upconv(nf * 4, nf * 2)
#         # state size. (nf*2) x 32 x 16
#         self.upc5 = dcgan_upconv(nf * 2, nf)
#         # state size. (nf) x 64 x 32
#         self.upc6 = nn.Sequential(
#                 nn.ConvTranspose2d(nf, nc, 4, 2, 1),
#                 nn.Tanh() # --> [-1, 1]
#                 #nn.Sigmoid()
#                 # state size. (nc) x 128 x 64
#                 )
#         self.apply(weights_init)

#     def forward(self, x):
#         d1 = self.upc1(x.view(-1, self.dim, 1, 1))
#         d2 = self.upc2(d1)
#         d3 = self.upc3(d2)
#         d4 = self.upc4(d3)
#         d5 = self.upc5(d4)
#         d6 = self.upc6(d5)
#         return d6

# class image_decoder_32(nn.Module):
#     def __init__(self, dim, nc=1):
#         super(image_decoder_32, self).__init__()
#         self.dim = dim
#         nf = 64
#         self.upc1 = nn.Sequential(
#                 # input is Z, going into a convolution
#                 nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
#                 nn.BatchNorm2d(nf * 8),
#                 nn.ReLU()
#                 )
#         # state size. (nf*8) x 4 x 4
#         self.upc2 = dcgan_upconv(nf * 8, nf * 8)
#         # state size. (nf*8) x 8 x 8
#         self.upc3 = dcgan_upconv(nf * 8, nf * 4)
#         # state size. (nf*4) x 16 x 16
#         self.upc4 = nn.Sequential(
#                 nn.ConvTranspose2d(nf, nc, 4, 2, 1),
#                 nn.Tanh() # --> [-1, 1]
#                 #nn.Sigmoid()
#                 # state size. (nc) x 32 x 32
#                 )
#         self.apply(weights_init)

#     def forward(self, x):
#         d1 = self.upc1(x.view(-1, self.dim, 1, 1))
#         d2 = self.upc2(d1)
#         d3 = self.upc3(d2)
#         d4 = self.upc4(d3)
#         d5 = self.upc5(d4)
#         d6 = self.upc6(d5)
#         return d6

class image_output_embed_128(nn.Module):
    def __init__(self, dim=128, nc=1):
        super(image_output_embed_128, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                #nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class discriminator_128(nn.Module):
    def __init__(self, dim=1, nc=1):
        super(discriminator_128, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            # [bs, 128, 128] --> [bs, 1, 128, 128]
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class audio_output_embed_128(nn.Module): # for Sub-URMP
    def __init__(self, dim=128, nc=1, dataset='SC09'):
        super(audio_output_embed_128, self).__init__()
        self.dim = dim
        nf = 64
        if dataset == 'SC09':
            # input is (nc) x 128 x 44
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        else:
            # input is (nc) x 128 x 22
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, (4, 3), (2, 1), (1, 1), bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        # state size. (nf) x 64 x 22
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 11
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 5
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 2
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 1
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, (4, 3), (1, 1), (0, 1)),
                #nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class classifier(nn.Module):
    def __init__(self, dim, n_class, use_bn=False):
        super(classifier, self).__init__()
        self.dim = dim
        self.n_class = n_class
        if use_bn:
            self.main = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),

                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        else:
            self.main = nn.Sequential(
                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, self.dim)
        out = self.main(x)
        return out
