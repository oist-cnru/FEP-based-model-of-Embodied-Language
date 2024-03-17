import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from .utils import mlp
import itertools

import math

# class CustomConvRNNCell(nn.Module):
#     """
#     Generate a convolutional LSTM cell
#     """
#     def __init__(self, topdown, input_size, hidden_size, kernel_size, stride, padding, dropout_rate=0.5,tau=0, input_dim=None, is_bottomup=True, is_lateral=True):
#         super(CustomConvLSTMCell, self).__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         # self.tau = tau
#         print("vision", tau)
#         self.topdown = topdown
#         self.is_lateral = is_lateral
#         self.is_bottomup = is_bottomup
#         if is_bottomup:
#             self.conv_bottomup = nn.Conv2d(input_size[0], 4 * hidden_size, kernel_size=kernel_size[0],
#                                            stride=stride[0], padding=padding[0])
#
#         # if bottomup == True:
#             # self.conv_bottomup = nn.Conv2d(input_size[0], 4*hidden_size, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
#         filter_size=None
#         if input_dim is not None:
#             #input_dim[0] layer -1 size - bottom up
#             #input_dim[1]  layer +1 size - top down
#             filter_size = math.floor((input_dim[0]+2*padding[0]-1*(kernel_size[0]-1) -1)/stride[0] + 1)
#
#         # self.conv_bottomup.bias.data.zero_()
#         if is_lateral:
#             self.conv_lateral_from_motor = nn.ConvTranspose2d(input_size[1], 4*hidden_size,
#                                                               kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
#         # self.conv_lateral_from_motor.bias.data.zero_()
#         if topdown:
#             output_padding = 0
#             if input_dim is not None:
#                 output_padding = filter_size - ((input_dim[1] - 1) * stride[0] - 2 * padding[0] + kernel_size[0])
#                 if output_padding>=stride[2]:
#                     #TODO in case one dimensional output, this fails:
#                     output_padding=0
#                     padding[2]=0
#
#             self.conv_topdown = nn.ConvTranspose2d(input_size[2], 4*hidden_size, kernel_size=kernel_size[2],
#                                                    stride=stride[2], padding=padding[2], output_padding=output_padding)
#
#         self.conv_lateral = nn.Conv2d(hidden_size, 4*hidden_size, kernel_size=5, stride=1, padding=2)
#
#         self.filter_size=filter_size
#         self.dropout = nn.Dropout(dropout_rate)
#     def forward(self, x, sx, bottomup=True):
#         # import pdb
#         # pdb.set_trace()
#         hx = sx
#         hy = self.conv_lateral(hx)
#         if bottomup:
#             hy = hy + self.conv_bottomup(x[0])  #+ self.conv_lateral_from_motor((x[1][:,:,None,None])) #
#         if self.is_lateral:
#             hy = hy + self.conv_lateral_from_motor((x[1][:, :, None, None]))
#         if self.topdown:
#             hy = hy + self.conv_topdown(x[2]) #+ self.conv_topdown(x[2])
#         layer_norm = nn.LayerNorm([hy.shape[1], hy.shape[2], hy.shape[3]]).to(hy.device)
#         hy = layer_norm(hy)
#         hy = F.tanh(hy)
#         hy = self.dropout(hy)
#
#         return hy

class CustomConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, topdown, input_size, hidden_size, kernel_size, stride, padding, dropout_rate=0.5,tau=0, input_dim=None, is_bottomup=True, is_lateral=True):
        super(CustomConvLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.tau = tau
        print("vision", tau)
        self.topdown = topdown
        self.is_lateral = is_lateral
        self.is_bottomup = is_bottomup
        if is_bottomup:
            self.conv_bottomup = nn.Conv2d(input_size[0], 4 * hidden_size, kernel_size=kernel_size[0],
                                           stride=stride[0], padding=padding[0])

        # if bottomup == True:
            # self.conv_bottomup = nn.Conv2d(input_size[0], 4*hidden_size, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        filter_size=None
        if input_dim is not None:
            #input_dim[0] layer -1 size - bottom up
            #input_dim[1]  layer +1 size - top down
            filter_size = math.floor((input_dim[0]+2*padding[0]-1*(kernel_size[0]-1) -1)/stride[0] + 1)

        # self.conv_bottomup.bias.data.zero_()
        if is_lateral:
            self.conv_lateral_from_motor = nn.ConvTranspose2d(input_size[1], 4*hidden_size,
                                                              kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        # self.conv_lateral_from_motor.bias.data.zero_()
        if topdown:
            output_padding = 0
            if input_dim is not None:
                output_padding = filter_size - ((input_dim[1] - 1) * stride[0] - 2 * padding[0] + kernel_size[0])
                if output_padding>=stride[2]:
                    #TODO in case one dimensional output, this fails:
                    output_padding=0
                    padding[2]=0

            self.conv_topdown = nn.ConvTranspose2d(input_size[2], 4*hidden_size, kernel_size=kernel_size[2],
                                                   stride=stride[2], padding=padding[2], output_padding=output_padding)

        self.conv_lateral = nn.Conv2d(hidden_size, 4*hidden_size, kernel_size=5, stride=1, padding=2)

        self.filter_size=filter_size
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, sx, bottomup=True):
        # import pdb
        # pdb.set_trace()
        hx, cx = sx
        gates = self.conv_lateral(hx)
        if bottomup:
            gates = gates + self.conv_bottomup(x[0])  #+ self.conv_lateral_from_motor((x[1][:,:,None,None])) #
        if self.is_lateral:
            gates = gates + self.conv_lateral_from_motor((x[1][:, :, None, None]))
        if self.topdown:
            gates = gates + self.conv_topdown(x[2]) #+ self.conv_topdown(x[2])
        layer_norm = nn.LayerNorm([gates.shape[1], gates.shape[2], gates.shape[3]]).to(gates.device)
        gates = layer_norm(gates)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy) #+ torch.randn_like(cy)
        hy = self.dropout(hy)
        return hy, cy, [ingate, forgetgate, cellgate, outgate]

class VisionStream(nn.Module):
    # TODO introduce dropout and some other ways to regularize the ConvLSTM
    def __init__(self, dropout_rate, num_inp_channels, num_out_channels, layers_spec, is_bottomup=True, is_lateral=True):
        super(VisionStream, self).__init__()
        self.bottomup = is_bottomup
        self.v_net = nn.ModuleList()
        num_layers = len(layers_spec)
        # print(self.v_net)
        for l in range(num_layers):
            is_topdown = l < (num_layers) -1
            zero_padding = int((layers_spec[l]['kernel_size']-1)/2)
            downscale_factor = layers_spec[l]['downscale_factor']
            bottomup_dim = layers_spec[l]['dim_bottomup']
            topdown_dim=-1

            #TODO dim and channels are mixed up
            if is_topdown:
                topdown_dim = layers_spec[l]['dim_topdown']
                topdown_channels = layers_spec[l+1]['num_filter']
            else:
                topdown_dim = 0
                topdown_channels = 0
            if l==0:
                bottomup=is_bottomup  #always true for the output layer
                layer_inp_channels = num_inp_channels  #bottom up input channels
            else:
                bottomup=is_bottomup
                layer_inp_channels = layers_spec[l-1]['num_filter']

            self.v_net.append(CustomConvLSTMCell(is_topdown,
                                                 [layer_inp_channels, layers_spec[l]['dim_lateral'],
                                                  topdown_channels],
                                                 layers_spec[l]['num_filter'],
                                                 kernel_size=[layers_spec[l]['kernel_size'],
                                                              layers_spec[l]['filter_size'], layers_spec[l]['kernel_size']],
                                                 stride=[downscale_factor, 1, downscale_factor],
                                                 padding=[zero_padding, 0, zero_padding],
                                                 input_dim=[bottomup_dim, topdown_dim], is_bottomup=bottomup,
                                                 is_lateral=is_lateral, dropout_rate=dropout_rate
                                                 ))
            if l==0:
                #add decoding output layer to first layer:
                stride=downscale_factor
                padding=zero_padding
                kernel_size=layers_spec[l]['kernel_size']
                output_padding = layers_spec[l]['dim_bottomup'] - ((layers_spec[l]['filter_size'] - 1) * stride - 2 * padding + kernel_size)
                self.v_decoding = nn.ConvTranspose2d(layers_spec[0]['num_filter'], num_out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)


    def forward(self, input, prev_state, lateral_inp, bottomup=True):

        layer_states = []
        layer_gates = []

        for layer_idx, (sx, topdown_input, incoming) in enumerate(
                itertools.zip_longest(prev_state, prev_state[1:], lateral_inp)):
            if layer_idx == 0:
                bottomup=self.bottomup
                if bottomup==False:
                    bottomup_input = []
                else:
                    bottomup_input = input

            else:
                bottomup=self.bottomup
                if bottomup==False:
                    bottomup=[]
                else:
                    bottomup_input = layer_states[-1][0]

            lateral_input = torch.cat(incoming, dim=1)

            if topdown_input is not None:
                input = [bottomup_input, lateral_input, topdown_input[0]] # bottomup_input, lateral_input,
            else:
                input = [bottomup_input, lateral_input] #bottomup_input, lateral_input

            h, c, gates = self.v_net[layer_idx](input, sx, bottomup=bottomup)
            layer_states.append([h, c])
            layer_gates.append(gates)

        out_pred = self.v_decoding(layer_states[0][0])


        return layer_states, out_pred, layer_gates

    def inspect_l0_memory(self, layer_states):
        decoded_mem_state = self.v_decoding(F.tanh(layer_states[0][1]))

        return decoded_mem_state

    def inspect_l1_memory(self, layer_states):
        l1_mem = layer_states[1][1]
        l0_state = self.v_net[0].conv_topdown(F.tanh(l1_mem))

        ingate, forgetgate, cellgate, outgate = l0_state.chunk(4, 1)

        cellgate[cellgate<0]=0
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        decoded_mem_state = self.v_decoding(F.tanh(cellgate))

        return decoded_mem_state
