import torch
from torch import nn
from torch.nn import functional as F

class LSTMCell(nn.Module):
	"""
	Generate a convolutional LSTM cell
	"""

	def __init__(self, input_size, hidden_size, dropout_rate=0.5,tau=0):
		super(LSTMCell, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.linear_ih = nn.Linear(input_size, 4*hidden_size)
		self.linear_hh = nn.Linear(hidden_size, 4*hidden_size)

		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, x, sx):
		hx, cx = sx
		gates = self.linear_ih(x) + self.linear_hh(hx)
		layer_norm = nn.LayerNorm([gates.shape[1]]).to(gates.device)
		gates = layer_norm(gates)
		ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
		ingate = F.sigmoid(ingate)
		forgetgate = F.sigmoid(forgetgate)
		cellgate = F.tanh(cellgate)
		outgate = F.sigmoid(outgate)

		cy = (forgetgate * cx) + (ingate * cellgate)
		hy = outgate * F.tanh(cy) #+ torch.randn_like(cy)
		hy = self.dropout(hy)
		return hy, cy, [ingate.detach(), forgetgate.detach(), cellgate.detach(), outgate.detach()]

class LSTMCell_PB(nn.Module):
	"""
	Generate a convolutional LSTM cell with PB
	"""

	def __init__(self, input_size, hidden_size, pb_size=3, dropout_rate=0.5, tau=0):
		super(LSTMCell_PB, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.pb_size = pb_size
		self.linear_ih = nn.Linear(input_size, 4*hidden_size)#.requires_grad_(False)
		self.linear_hh = nn.Linear(hidden_size, 4*hidden_size)#.requires_grad_(False)
		self.linear_pbh = nn.Linear(pb_size, 4*hidden_size)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x, sx, pb):

		hx, cx = sx
		gates = self.linear_ih(x) + self.linear_hh(hx) + self.linear_pbh(pb)
		layer_norm = nn.LayerNorm([gates.shape[1]]).to(gates.device)
		gates = layer_norm(gates)
		ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
		ingate = F.sigmoid(ingate)
		forgetgate = F.sigmoid(forgetgate)
		cellgate = F.tanh(cellgate)
		outgate = F.sigmoid(outgate)

		cy = (forgetgate * cx) + (ingate * cellgate)
		hy = outgate * F.tanh(cy) #+ torch.randn_like(cy)
		hy = self.dropout(hy)
		return hy, cy, [ingate.detach(), forgetgate.detach(), cellgate.detach(), outgate.detach()]

class PVMTRNNCell(nn.Module):
	"""
	Generate a PVMTRNN cell
	"""

	def __init__(self, hidden_size, z_size,  pb_size=0, tau=1, input_size=None, dropout_rate=0.0):
		super(PVMTRNNCell, self).__init__()
		self.dropout = nn.Dropout(dropout_rate)
		self.tau = tau
		if input_size != 0:   # includes top down
			self.fc_ih = nn.Linear(input_size, hidden_size)
		self.fc_zh = nn.Linear(z_size, hidden_size)
		self.fc_hmup = nn.Linear(hidden_size, z_size)
		self.fc_hsigmap = nn.Linear(hidden_size, z_size)
		self.fc_hh = nn.Linear(hidden_size, hidden_size)
		# if top_h_size!=0:
		# 	self.fc_hdh = nn.Linear(top_h_size, hidden_size)
	def forward(self, x, h, mu_q, logvar_q, mu_p_i, logvar_p_i, gen_prior=False, step=0):
		#compute prior and posterior
		torch.manual_seed(0)
		# x includes top down input if any
		if step == 0:
			mu_p = F.tanh(mu_p_i)
			sigma_p = torch.exp(logvar_p_i)
		else:
			mu_p = F.tanh(self.fc_hmup(h))
			sigma_p = torch.exp(self.fc_hsigmap(h))
		mu_q = F.tanh(mu_q)
		sigma_q = torch.exp(logvar_q)
		zq = mu_q + sigma_q*torch.randn(sigma_q.size(), device=sigma_q.device)
		zp = mu_p + sigma_p*torch.randn(sigma_p.size(), device=sigma_p.device)
		# print("zshape={}".format(zq.shape))
		if gen_prior: # generating from prior
			if x is not None:
				hx = (1 - 1 / self.tau) * h + (self.fc_ih(x) + self.fc_hh(h) + self.fc_zh(zp)) / (self.tau)
			else:
				hx = (1 - 1 / self.tau) * h + (self.fc_hh(h) + self.fc_zh(zp)) / (self.tau)
		else: # generating from posterior
			if x is not None:
				hx = (1 - 1 / self.tau) * h + (self.fc_ih(x) + self.fc_hh(h) + self.fc_zh(zq)) / (self.tau)
			else:
				hx = (1 - 1 / self.tau) * h + (self.fc_hh(h) + self.fc_zh(zq)) / (self.tau)

		d = F.tanh(hx)
		d = self.dropout(d)
		return d, mu_p, sigma_p, zq


def mlp(in_size, emb_dim, out_size, layer_norm=True, activate_final=None):
    modules = []

    modules.append(nn.Linear(in_size, emb_dim))
    if layer_norm:
        modules.append(nn.LayerNorm(emb_dim))
    modules.append(nn.ReLU(inplace=True))

    modules.append(nn.Linear(emb_dim, out_size))
    # if layer_norm:
    #     modules.append(nn.LayerNorm(out_size))
    if activate_final is not None:
        modules.append(activate_final)

    return nn.Sequential(*modules)

def masked_loss(criterion, input, target, mask):
	"""
	compute loss for masked data
	does not work with cross entropy loss function
	"""
	# input [NxTxC]:
	# mask [NxT]:
	# assert criterion.size_average is False, 'size_average should be set to False'
	# assert criterion.reduce is False, 'reduce should be set to False'
	normalized_mask = F.normalize(mask, p=1, dim=1)
	loss = criterion(input, target)
	singleton_expansion = [1 for _ in range(loss.dim() - 2)]
	loss = loss * normalized_mask.view(*normalized_mask.size(), *singleton_expansion)
	loss = loss.sum(1)
	loss = loss.mean()

	return loss

def kl_criterion(mu1, logvar1, mu2, logvar2):
	"""
	compute KL-Divergence
	"""
	sigma1 = logvar1.mul(0.5).exp()
	sigma2 = logvar2.mul(0.5).exp() 
	loss = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
	loss = loss.mean(0)
	loss = loss.sum()
	return loss

def kl_fixed_logvar_criterion(mu1, mu2):

	loss = ((mu1 - mu2)**2)/2
	loss = loss.mean(0)
	loss = loss.sum()

	return loss

