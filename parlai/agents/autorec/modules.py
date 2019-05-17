import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def init_weight(linear_module):
    stdv = 1. / math.sqrt(linear_module.weight.size(1))
    linear_module.weight.data.uniform_(-stdv, stdv)
    if linear_module.bias is not None:
        linear_module.bias.data.uniform_(-stdv, stdv)

class UserEncoder(nn.Module):
    def __init__(self, layer_sizes, n_movies, f):
        """

        :param layer_sizes: list giving the size of each layer
        :param n_movies:
        :param f:
        """
        super(UserEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=n_movies, out_features=layer_sizes[0]) if i == 0
                                     else nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
                                     for i in range(len(layer_sizes))])

        if f == 'identity':
            self.f = lambda x: x
        elif f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(f))

    def forward(self, input, raw_last_layer=False):
        for (i, layer) in enumerate(self.layers):
            if raw_last_layer and i == len(self.layers) - 1:
                # do not apply activation to last layer
                input = layer(input)
            else:
                input = self.f(layer(input))
        return input


class AutoRec(nn.Module):
    """
    User-based Autoencoder for Collaborative Filtering
    """

    def __init__(self,
                 n_movies,
                 params,
                 resume=None):
        super(AutoRec, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.n_movies = n_movies
        self.layer_sizes = params['layer_sizes']
        if params['g'] == 'identity':
            self.g = lambda x: x
        elif params['g'] == 'sigmoid':
            self.g = nn.Sigmoid()
        elif params['g'] == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(params['g']))

        if resume is not None:
            # Load pretrained model, keeping only the first n_movies. (see merge_indexes function in match_movies)
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            # get layer sizes from checkpoint if saved
            if "layer_sizes" in checkpoint:
                self.layer_sizes = checkpoint["layer_sizes"]
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)
            model_dict = self.state_dict()
            # load all weights except for the weights of the first layer and the decoder
            model_dict.update({k: v for k, v in checkpoint['state_dict'].items()
                               if k != "encoder.layers.0.weight" and "decoder" not in k})
            # load first layer and decoder: assume the movies to keep are the n_movies first movies
            encoder0weight = checkpoint["state_dict"]["encoder.layers.0.weight"][:, :self.n_movies]
            decoderweight = checkpoint["state_dict"]["decoder.weight"][:self.n_movies, :]
            decoderbias = checkpoint["state_dict"]["decoder.bias"][:self.n_movies]
            # If checkpoint has fewer movies than the model, append zeros (no actual recommendations for new movies)
            # (When using an updated movie list)
            if encoder0weight.shape[1] < self.n_movies:
                tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
                encoder0weight = torch.cat((
                    encoder0weight,
                    torch.zeros(encoder0weight.shape[0], self.n_movies - encoder0weight.shape[1], out=tt())), dim=1)
                decoderweight = torch.cat((
                    decoderweight,
                    torch.zeros(self.n_movies - decoderweight.shape[0], decoderweight.shape[1], out=tt())), dim=0)
                decoderbias = torch.cat((
                    decoderbias, torch.zeros(self.n_movies - decoderbias.shape[0], out=tt())), dim=0)
            model_dict.update({
                "encoder.layers.0.weight": encoder0weight,
                "decoder.weight": decoderweight,
                "decoder.bias": decoderbias,
            })
            self.load_state_dict(model_dict)
        else:
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)

        if self.cuda_available:
            self.cuda()

    def forward(self, input, additional_context=None, range01=True):
        """

        :param input: (batch, n_movies)
        :param additional_context: potential information to add to user representation (batch, user_rep_size)
        :param range01: If true, apply sigmoid to the output
        :return: output recommendations (batch, n_movies)
        """
        # get user representation
        encoded = self.encoder(input, raw_last_layer=True)
        # eventually use additional context
        if additional_context is not None:
            encoded = self.encoder.f(encoded + additional_context)
        else:
            encoded = self.encoder.f(encoded)
        # decode to get movie recommendations
        if range01:
            return self.g(self.decoder(encoded))
        else:
            return self.decoder(encoded)

    def evaluate(self, batch_loader, criterion, subset, batch_input):
        """
        Evaluate function
        :param batch_loader:
        :param criterion:
        :param batch_input:
        :param subset: in {"test", "valid", "train"}. Susbet on which to evaluate
        :return: the mean loss.
        """
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        for _ in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset=subset, batch_input=batch_input)
            if self.cuda_available:
                batch["input"] = batch["input"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            output = self.forward(batch["input"])
            loss = criterion(output, batch["target"])
            losses.append(loss.item())
        # normalize loss and reset nb of ratings observed
        final_loss = criterion.normalize_loss_reset(np.sum(losses))
        print("{} loss with input={} : {}".format(subset, batch_input, final_loss))
        self.train()
        return final_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        # Sum of losses
        self.mse_loss = nn.MSELoss(size_average=False)
        # Keep track of number of observer targets to normalize loss
        self.nb_observed_targets = 0

    def forward(self, input, target):
        # only consider the observed targets
        mask = (target != -1)
        observed_input = torch.masked_select(input, mask)
        observed_target = torch.masked_select(target, mask)
        # increment nb of observed targets
        self.nb_observed_targets += len(observed_target)
        loss = self.mse_loss(observed_input, observed_target)
        return loss

    def normalize_loss_reset(self, loss):
        """
        returns loss divided by nb of observed targets, and reset nb_observed_targets to 0
        :param loss: Total summed loss
        :return: mean loss
        """
        if self.nb_observed_targets == 0:
            raise ValueError(
                "Nb observed targets was 0. Please evaluate some examples before calling normalize_loss_reset")
        n_loss = loss / self.nb_observed_targets
        self.nb_observed_targets = 0
        return n_loss
