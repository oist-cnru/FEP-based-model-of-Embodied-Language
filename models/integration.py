import torch
from torch import nn
from torch.nn import functional as F

import itertools
from .utils import LSTMCell
from .utils import PVMTRNNCell
# from .utils import PVMTRNNPBCell



class Integration_bind(nn.Module):
	"""
	Integration network with language binding
	"""
	def __init__(self, in_sizes, layers, dropout_rate=0.5, is_UG=False):
		super(Integration_bind, self).__init__()
		# self.hid_size = hid_size
		self.num_layers = len(layers)
		self.d_size = [layers[l]['hid_size'] for l in range(len(layers))]
		self.z_size = [layers[l]['z_size'] for l in range(len(layers))]

		self.tau = [layers[l]['tau'] for l in range(len(layers))]
		self.integration_net = nn.ModuleList()

		for l in range(len(layers)):
			layer = PVMTRNNCell(self.d_size[l], z_size=self.z_size[l],
										tau=self.tau[l], input_size=in_sizes[l], dropout_rate=dropout_rate)

			self.integration_net.append(layer)

	def forward(self, incomings, sxs, mu_q, sigma_q, mu_p_i, logvar_p_i, gen_prior=False, step=0): #
		# assert len(sxs) == len(incomings), ''
		zs = []
		ss = []
		gates = []
		# start = 0
		mu_p, sigma_p = [], []
		for layer_idx, (sx, topdown_sx, incoming) in enumerate(itertools.zip_longest(sxs, sxs[1:], incomings)):
			# print("incoming shape", str(incoming.shape))
			if layer_idx == len(sxs) - 1:
				input = [*incoming]
				if len(input) != 0:
					input = torch.cat(input, dim=1)
				else:
					input = None
				h, mu_p_, sigma_p_, z = self.integration_net[layer_idx](input, sx[0], mu_q[layer_idx][:, step],
																		sigma_q[layer_idx][:, step], mu_p_i[layer_idx],
																		logvar_p_i[layer_idx],
																		gen_prior=gen_prior, step=step)
				zs.append(z)
				mu_p.append(mu_p_)
				sigma_p.append(sigma_p_)
			else:
				input = None
				h, mu_p_, sigma_p_, z = self.integration_net[layer_idx](input, sx[0], mu_q[layer_idx][:, step],
																		sigma_q[layer_idx][:, step], mu_p_i[layer_idx],
																		logvar_p_i[layer_idx],
																		gen_prior=gen_prior, step=step)
				zs.append(z)
				mu_p.append(mu_p_)
				sigma_p.append(sigma_p_)
			ss.append([h, torch.zeros_like(h)])



		return ss, mu_p, sigma_p, gates, zs
