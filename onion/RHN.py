import torch.nn as nn
import onion.util
import numbers
import numpy as np
import torch.nn.functional as F

floatX = 'float32'

def Linear(nn.Module):

    def __init__(self, in_size, out_size, bias_init=None, init_scale=0.04):
        super(Linear, self).__init__()
        util.autoassign(locals())
        self.layer = nn.Linear(in_size, out_size, bias=bias_init is not None)
        self.layer.weight.data.uniform_(-init_scale, init_scale)
        if isinstance(self.bias_init, numbers.Number):
            self.layer.bias.data.uniform_(bias_init)
        elif bias_init == 'uniform':
            self.layer.bias.data.uniform_(-init_scale, init_scale)
        else:
            raise AssertionError('unsupported init_scheme')

    def forward(self, x):
        return self.layer(x)


class RHN(nn.Module):
    """Recurrent Highway Network. Based on
    https://arxiv.org/abs/1607.03474 and
    https://github.com/julian121266/RecurrentHighwayNetworks.

    """
    def __init__(self, size_in, size, recur_depth=1, drop_i=0.75 , drop_s=0.25,
                 init_T_bias=-2.0, init_H_bias='uniform', tied_noise=True, init_scale=0.04, seed=1):
        super(RHN, self).__init__()
        util.autoassign(locals())
        hidden_size = self.size
        self.LinearH = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_H_bias)
        self.LinearT = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_T_bias)
        self.recurH = nn.ModuleList()
        self.recurT = nn.ModuleList()
        for l in range(self.recur_depth):
            if l == 0:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size))
            else:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_H_bias))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_T_bias))

    def apply_dropout(self, x, noise):
        if self..training:
            return noise * x
        else:
            return x

    def get_dropout_noise(self, shape, dropout_p):
        keep_p = 1 - dropout_p
        noise = (1. / keep_p) * torch.bernoulli(torch.zeros(shape) + keep_p)
        return noise

    def step(self, i_for_H_t, i_for_T_t, h_tm1, noise_s):
        tanh, sigm = F.tanh, F.sigmoid
        noise_s_for_H = noise_s if self.tied_noise else noise_s[0]
        noise_s_for_T = noise_s if self.tied_noise else noise_s[1]

        hidden_size = self.size
        s_lm1 = h_tm1
        for l in range(self.recur_depth):
            s_lm1_for_H = self.apply_dropout(s_lm1, noise_s_for_H)
            s_lm1_for_T = self.apply_dropout(s_lm1, noise_s_for_T)
            if l == 0:
                # On the first micro-timestep of each timestep we already have bias
                # terms summed into i_for_H_t and into i_for_T_t.
                H = tanh(i_for_H_t + self.recurH[l](s_lm1_for_H))
                T = sigm(i_for_T_t + self.recurT[l](s_lm1_for_T))
            else:
                H = tanh(self.recurH[l](s_lm1_for_H))
                T = sigm(self.recurT[l](s_lm1_for_T))
            s_l = (H - s_lm1) * T + s_lm1
            s_lm1 = s_l

        y_t = s_l
        return y_t

    def forward(self, h0, seq, repeat_h0=1):
        inputs = seq.dimshuffle((1,0,2))
        (_seq_size, batch_size, _) = inputs.size()
        hidden_size = self.size
        # We first compute the linear transformation of the inputs over all timesteps.
        # This is done outside of scan() in order to speed up computation.
        # The result is then fed into scan()'s step function, one timestep at a time.
        noise_i_for_H = self.get_dropout_noise((batch_size, self.size_in), self.drop_i)
        noise_i_for_T = self.get_dropout_noise((batch_size, self.size_in), self.drop_i) if not self.tied_noise else noise_i_for_H

        i_for_H = self.apply_dropout(inputs, noise_i_for_H)
        i_for_T = self.apply_dropout(inputs, noise_i_for_T)

        i_for_H = self.LinearH(i_for_H)
        i_for_T = self.LinearT(i_for_T)

        # Dropout noise for recurrent hidden state.
        noise_s = self.get_dropout_noise((batch_size, hidden_size), self.drop_s)
        if not self.tied_noise:
          noise_s = tt.stack(noise_s, self.get_dropout_noise((batch_size, hidden_size), self.drop_s))
        #TODO  replace SCAN with a LOOP
        #H0 = tt.repeat(h0, inputs.shape[1], axis=0) if repeat_h0 else h0
        H0 = h0.expand(inputs.size(1)) if repeat_h0 else h0
        out, _ = theano.scan(self.step,
                             sequences=[i_for_H, i_for_T],
                             outputs_info=[H0],
                             non_sequences = [noise_s])
        return out.dimshuffle((1, 0, 2))
