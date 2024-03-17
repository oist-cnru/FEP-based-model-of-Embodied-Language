import torch
from torch import nn
from torch.nn import functional as F
from .vision import Vision, VisionStream
from .motor import Motor
from .integration import Integration

from .utils import mlp

class Model(nn.Module):
	def __init__(self, num_context_frames, intention_size, vision_args, motor_args, integration_args):
		super(Model, self).__init__()
		self.num_context_frames = num_context_frames
		self.intention_size = intention_size

		v_enc_dim = 4*4*16*4 #vision_args['enc_dim']
		v_hid_size = vision_args['hid_sizes']

		self.num_joints = motor_args['num_joints']
		self.joint_enc_dim = motor_args['joint_enc_dim']

		m_enc_dim = motor_args['num_joints'] * motor_args['joint_enc_dim']
		m_hid_size = motor_args['hid_sizes']

		i_hid_size = integration_args['hid_sizes']

		v_in_sizes = (	
			2 * v_enc_dim + v_hid_size[1] + m_hid_size[0], 
			v_hid_size[0] + v_hid_size[2] + m_hid_size[1],
			v_hid_size[1] + i_hid_size[0] + m_hid_size[2],
		)

		self.vision_net = Vision(in_sizes=v_in_sizes, **vision_args)

		m_in_sizes = (	
			m_enc_dim + 3 + m_hid_size[1] + 16*16*16,#v_hid_size[0], 
			m_hid_size[0] + m_hid_size[2] + 32*8*8,#v_hid_size[1],
			m_hid_size[1] + i_hid_size[0] + 64*4*4,#v_hid_size[2],
		)
		# import pdb
		# pdb.set_trace()
		self.md_net = Motor(in_sizes=m_in_sizes, **motor_args)

		m_in_sizes = (	
			m_enc_dim + 3 + m_hid_size[1] + 16*16*16,#v_hid_size[0], 
			m_hid_size[0] + m_hid_size[2] + 32*8*8,#v_hid_size[1],
			m_hid_size[1] + i_hid_size[0] + 64*4*4,#v_hid_size[2],
		)
		self.mv_net = Motor(in_sizes=m_in_sizes, **motor_args)

		# integration net receives the highest vision and motor outputs & intention as inputs
		i_in_sizes = (2*64*4*4 + 2*m_hid_size[2], )
		# i_in_sizes = (v_hid_size + m_hid_size + intention_size, )
		self.integration_net = Integration(in_sizes=i_in_sizes, **integration_args)

		self.m_enc_dim = m_enc_dim
		self.fc_m_pred = nn.Sequential(
			nn.Linear(2 * m_hid_size[0], 256),
			nn.LayerNorm(256),
			nn.ReLU(inplace=True),
			nn.Linear(256, m_enc_dim)
		)

		self.fc_att_where = nn.Sequential(
			nn.Linear(512, 256),
			nn.LayerNorm(256),
			nn.ReLU(inplace=True),
			nn.Linear(256,3)
		)


		self.fc_att_where[-1].weight.data.fill_(0)
		self.fc_att_where[-1].bias.data = torch.FloatTensor([4., 0, 0])


	def forward(self, vision, motor, cell_mu, cell_logvar):

		n = vision.size(0)
		device = vision.device
		
		
		# import pdb
		# pdb.set_trace()

		# hidden = hidden_mu + torch.exp(0.5 * hidden_logvar) * hidden_logvar.new(hidden_logvar.size()).normal_()
		
		# print(hidden_logvar.mean())
		cell = cell_mu + torch.exp(0.5 * cell_logvar) * cell_logvar.new(cell_logvar.size()).normal_()
		hidden = F.tanh(cell)
		print("posterior", cell_mu.max().item(), cell_mu.min().item(), cell_logvar.max().item(), cell_logvar.min().item())


		v_init_hidden, md_init_hidden, mv_init_hidden, i_init_hidden = hidden.split([2*(16*16*16+32*8*8+64*4*4), 448, 448, 512], dim=1)
		v_init_cell, md_init_cell, mv_init_cell, i_init_cell = cell.split([2*(16*16*16+32*8*8+64*4*4), 448, 448, 512], dim=1)

		
		# v_init_state = [state for state in zip(v_init_hidden.split([512, 256, 128, 64], dim=1), v_init_cell.split([512, 256, 128, 64], dim=1))]
		md_init_state = [state for state in zip(md_init_hidden.split([256, 128, 64], dim=1), md_init_cell.split([256, 128, 64], dim=1))]
		mv_init_state = [state for state in zip(mv_init_hidden.split([256, 128, 64], dim=1), mv_init_cell.split([256, 128, 64], dim=1))]
		i_init_state = [(i_init_hidden, i_init_cell)]



		# convert from 'batch' (N) first to 'seq_len' (T) first 
		vision = vision.transpose(0,1)
		motor = motor.transpose(0,1)

		# set initial state to zero values
		# m_init_state = [init_states[:,:512], init_states[:,512:512+512]] #self.vision_net.zero_init_state(n, device)
		# v_init_state = [init_states[:,512+512:512+512+512], init_states[:,512+512+512:512+512+512+512]]#self.motor_net.zero_init_state(n, device)
		# # i_init_state = self.integration_net.zero_init_state(n, device)
		# i_init_state = [init_states[:,512+512+512+512:]] 

		md_states, mv_states, i_states = [], [], []

		# v_states.extend(v_init_state)
		md_states.extend(md_init_state)
		mv_states.extend(mv_init_state)
		i_states.extend(i_init_state)




		# column_init_hidden = hidden[:,2*(512+256+128+64)+64:]
		# column_init_cell = cell[:,2*(512+256+128+64)+64:]

		cv_init_hidden, pv_init_hidden = v_init_hidden.split([16*16*16+32*8*8+64*4*4,16*16*16+32*8*8+64*4*4], dim=1)
		cv_init_cell, pv_init_cell = v_init_cell.split([16*16*16+32*8*8+64*4*4,16*16*16+32*8*8+64*4*4], dim=1)

		v_states = [[[
				(cv_init_hidden[:,:16*16*16].view(n, 16, 16, 16), cv_init_cell[:,:16*16*16].view(n, 16, 16, 16)), 
				(cv_init_hidden[:,16*16*16:16*16*16+32*8*8].view(n, 32, 8, 8), cv_init_cell[:,16*16*16:16*16*16+32*8*8].view(n, 32, 8, 8)), 
				(cv_init_hidden[:,16*16*16+32*8*8:].view(n, 64, 4, 4), cv_init_cell[:,16*16*16+32*8*8:].view(n, 64, 4, 4))
			],
			[
				(pv_init_hidden[:,:16*16*16].view(n, 16, 16, 16), pv_init_cell[:,:16*16*16].view(n, 16, 16, 16)), 
				(pv_init_hidden[:,16*16*16:16*16*16+32*8*8].view(n, 32, 8, 8), pv_init_cell[:,16*16*16:16*16*16+32*8*8].view(n, 32, 8, 8)), 
				(pv_init_hidden[:,16*16*16+32*8*8:].view(n, 64, 4, 4), pv_init_cell[:,16*16*16+32*8*8:].view(n, 64, 4, 4)) 
			]
		]]

		canvas = None

		cv_predictions = []
		pv_predictions = []
		rv_predictions = []
		m_predictions = []
		backgrounds = []

		attention_wheres = []

		mus, logvars = [], []



		for step, (ext_v, ext_m) in enumerate(zip(vision, motor)):

			# if step > 5:
			# 	v = 0.1 * v + 0.9 * v_predictions[-1]

			# import pdb
			# pdb.set_trace()
			# closed-loop in test phase
			if step > self.num_context_frames - 1:
				if self.training:
					# v = ext_v 
					# m = ext_m 
					feedback_ratio = 0.9

					v = (1. - feedback_ratio) * ext_v + feedback_ratio * rv_predictions[-1]
					m = (1. - feedback_ratio) * ext_m + feedback_ratio * torch.exp(m_predictions[-1])
				else:
					# v = ext_v 
					# m = ext_m 

					v = rv_predictions[-1]
					m = torch.exp(m_predictions[-1])
			else:
				v = ext_v
				m = ext_m


			pv_sxs = (v_states[-1][1][-3], v_states[-1][1][-2], v_states[-1][1][-1])
			cv_sxs = (v_states[-1][0][-3], v_states[-1][0][-2], v_states[-1][0][-1])
			md_sxs = (md_states[-3], md_states[-2], md_states[-1])
			mv_sxs = (mv_states[-3], mv_states[-2], mv_states[-1])
			i_sxs = (i_states[-1], )

			# motor processing
			if step == 0:
				read_where = self.fc_att_where(torch.cat([md_sxs[0][0], mv_sxs[0][0]], dim=1))
				attention_wheres.append(read_where)

			md_incomings = (
				(pv_sxs[-3][0].view(n,-1), attention_wheres[-1],), 
				(pv_sxs[-2][0].view(n,-1), ), 
				(pv_sxs[-1][0].view(n,-1), i_sxs[-1][0])
			)
			md_ss = self.md_net(m, md_incomings, md_sxs)

			mv_incomings = (
				(cv_sxs[-3][0].view(n,-1), attention_wheres[-1],), 
				(cv_sxs[-2][0].view(n,-1), ), 
				(cv_sxs[-1][0].view(n,-1), i_sxs[-1][0])
			)
			mv_ss = self.mv_net(m, mv_incomings, mv_sxs)


			m_pred = self.fc_m_pred(torch.cat([md_ss[0][0], mv_ss[0][0]], dim=1)) #self.fc_m_decoding(F.relu(self.fc_m_pred(hs[1])))
			
			# reshape motor prediction for applying softmax or log softmax to each joint
			m_pred = m_pred.view(-1, self.num_joints, self.joint_enc_dim)
			m_pred = F.log_softmax(m_pred, dim=2)
			m_pred = m_pred.view(-1, self.m_enc_dim)


			write_where = self.fc_att_where(torch.cat([md_ss[0][0], mv_ss[0][0]], dim=1))
			attention_wheres.append(write_where)

			# vision processing
			vd_incomings = (
				(md_sxs[-3][0], ), 
				(md_sxs[-2][0], ), 
				(md_sxs[-1][0], i_sxs[-1][0])
			)
			vv_incomings = (
				(mv_sxs[-3][0], ), 
				(mv_sxs[-2][0], ), 
				(mv_sxs[-1][0], i_sxs[-1][0])
			)
			
			if step > self.num_context_frames - 2:
				if step == self.num_context_frames - 1:
					canvas = ext_v
					prev_canvas = ext_v
				v_ss, v_pred, canvas, prev_canvas = self.vision_net(v, vd_incomings, vv_incomings, v_states[-1], attention_wheres[-2], write_where, i_sxs[-1][0], canvas, prev_canvas)
			else:
				v_ss, v_pred, _, _ = self.vision_net(v, vd_incomings, vv_incomings, v_states[-1], attention_wheres[-2], write_where, i_sxs[-1][0], canvas, prev_canvas)

			# import pdb
			# pdb.set_trace()
			# integration of vision, motor, and intention
			i_incomings = (
				(v_ss[0][-1][0].view(n,-1), v_ss[1][-1][0].view(n,-1), md_ss[-1][0], mv_ss[-1][0]), #(v_hs[-1], m_hs[-1], sampled_intention),
			)
			# i_incomings = (
			# 	(v_hs[-1], m_hs[-1], sampled_intention),
			# )
			i_ss = self.integration_net(i_incomings, i_sxs)

			md_states.extend(md_ss)
			mv_states.extend(mv_ss)
			i_states.extend(i_ss)

			v_states.append(v_ss)

			# mus.append(mu)
			# logvars.append(logvar)

			if step > self.num_context_frames - 2:
				rv_pred, cv_pred, pv_pred, background = v_pred
				cv_predictions.append(cv_pred)
				pv_predictions.append(pv_pred)
				rv_predictions.append(rv_pred)
				m_predictions.append(m_pred)
				backgrounds.append(background)



		# outputs are formed in batch first
		cv_predictions = torch.stack(cv_predictions, dim=1)
		pv_predictions = torch.stack(pv_predictions, dim=1)
		rv_predictions = torch.stack(rv_predictions, dim=1)
		m_predictions = torch.stack(m_predictions, dim=1)
		backgrounds = torch.stack(backgrounds, dim=1)

		# import pdb
		# pdb.set_trace()
		attention_wheres = torch.stack(attention_wheres[self.num_context_frames:], dim=1)

		return m_predictions, rv_predictions, cv_predictions, pv_predictions, attention_wheres, backgrounds


