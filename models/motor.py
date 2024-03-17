import torch
from torch import nn
from torch.nn import functional as F

from .utils import mlp, LSTMCell

import itertools

class Motor_vision(nn.Module):
	def __init__(self, num_joints, joint_enc_dim, in_sizes,
				 readout_hid_size, layers, dropout_rate=0.5, is_softmax=True, is_bottomup=True, is_input=True, is_lateral=True):
		super(Motor_vision, self).__init__()
		self.is_bottomup = is_bottomup
		self.is_input = is_input
		self.is_lateral = is_lateral
		self.num_joints = num_joints
		self.joint_enc_dim = joint_enc_dim
		self.enc_dim = num_joints * joint_enc_dim
		self.num_layers = len(in_sizes)
		self.motor_net = nn.ModuleList()
		for l in range(len(layers)):
			layer = LSTMCell(in_sizes[l], layers[l]['hid_size'], dropout_rate=dropout_rate)
			self.motor_net.append(layer)


	def forward(self, x, incomings, sxs):
		assert len(sxs) == len(incomings), ''

		ss = []  # [self.fc_m_encoding(x)]
		gates = []

		for layer_idx, (sx, topdown_sx, incoming) in enumerate(itertools.zip_longest(sxs, sxs[1:], incomings)):
			if self.is_input==False:
				x = None
			if layer_idx == 0:
				if x is not None:
					input = [x, *incoming]
				else:
					input = [*incoming]
			elif self.is_bottomup:
				input = [ss[-1][0], *incoming]
			else:
				input = [*incoming]

			if topdown_sx is not None:
				input.append(topdown_sx[0])
			input = torch.cat(input, dim=1)
			h, c, gate = self.motor_net[layer_idx](input, sx)
			ss.append([h, c])
			gates.append(gate)
		return ss, gates
