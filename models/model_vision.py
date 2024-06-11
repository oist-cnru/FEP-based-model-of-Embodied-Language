import torch
from torch import nn
from torch.nn import functional as F
from .vision import VisionStream
from .motor import Motor_vision
from .integration import Integration_bind
from .language import Language

import itertools
import numpy as np
from .utils import mlp
from .utils import kl_criterion

torch.backends.cudnn.benchmark == True

class SpatialTransformer(nn.Module):
    def __init__(self, outDim, inpDim):
        super(SpatialTransformer, self).__init__()
        self.register_buffer('expansion_indices',
                             torch.LongTensor([1, 0, 2, 0, 1, 3]))  # for conversion of params to affine matrix
        self.att_size = outDim
        self.rec_size = inpDim

    # TODO attention functionality:
    def where_inv(self, where):
        # Take a batch of where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,dx,dy] -> [1/s,-dx/s,-dy/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = where.size(0)
        out = torch.cat((where.new_ones((n, 1)), -where[:, 1:]), dim=1)
        # Divide all entries by the scale.
        out = out / where[:, 0:1]  # torch.index_select(where, 1, self.scale_indices)
        return out

    def where_to_center(self, where):
        # return the center of the window normalized to 0..1
        px = 1 / (2 * where[:, :, 0].detach()) + (where[:, :, 1] + 1) * 0.5 * (1 - 1 / where[:, :, 0].detach())
        py = 1 / (2 * where[:, :, 0].detach()) + (where[:, :, 2] + 1) * 0.5 * (1 - 1 / where[:, :, 0].detach())
        return torch.stack([1 - px, 1 - py], dim=-1)

    def expand_where(self, where):
        # Takes 4-dimensional vectors, and massages them into 2x3 matrices with elements like so:
        # [s,dx,dy] -> [[s,0,dx],
        #             	[0,s,dy]]

        # where [Nx3]: 		parameters for attention
        assert where.size(1) == 3, '3D-only'
        n = where.size(0)
        out = torch.cat((where.new_zeros((n, 1)), where), dim=1)
        return torch.index_select(out, 1, self.expansion_indices).view(n, 2, 3)

    def image_to_window(self, image, where):
        # image [NxCxHxW]: 	input image
        # where [Nx4]: 		parameters for attention
        assert image.size(0) == where.size(0), 'batch sizes of image and where are not matched'
        n, c, h, w = image.size()
        theta = self.expand_where(self.where_inv(where))
        grid = F.affine_grid(theta, torch.Size((n, c, self.att_size, self.att_size))) #, align_corners=True
        out = F.grid_sample(image.view(n, c, h, w), grid) #, align_corners=True
        return out

    def window_to_image(self, window, where, image_size):
        assert window.size(2) == window.size(3), 'window should be a square image'
        n, c, window_size, window_size = window.size()
        assert window_size == self.att_size, 'window size is not mathced with attention size'
        theta = self.expand_where(where)
        grid = F.affine_grid(theta, torch.Size((n, c, image_size[0], image_size[1]))) #, align_corners=True
        out = F.grid_sample(window.view(n, c, self.att_size, self.att_size), grid) #, align_corners=True
        return out

    def forward(self):
        return
    def getInit(self):
        return torch.FloatTensor([2.0, 0.0, 0.0])

class SpatialTransformer_2Dscaling(nn.Module):
    def __init__(self, outDim, inpDim):
        super(SpatialTransformer_2Dscaling, self).__init__()

        self.register_buffer('expansion_indices',
                             torch.LongTensor([1, 0, 3, 0, 2, 4]))  # for conversion of params to affine matrix

        self.att_size = outDim
        self.rec_size = inpDim

    # TODO attention functionality:
    def where_inv(self, where):
        # Take a batch of where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,dx,dy] -> [1/s,-dx/s,-dy/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = where.size(0)
        out = torch.cat((where.new_ones((n, 2)),  -where[:, 2:]), dim=1)
        out_div = torch.cat((where[:, 0:1],where[:, 1:2],where[:, 0:1],where[:, 1:2]), dim=1)
        # Divide all entries by the scale.
        out = out / out_div  # torch.index_select(where, 1, self.scale_indices)
        return out

    def where_to_center(self, where):
        # return the center of the window normalized to 0..1
        px = 1 / (2 * where[:, :, 0].detach()) + (where[:, :, 2] + 1) * 0.5 * (1 - 1 / where[:, :, 0].detach())
        py = 1 / (2 * where[:, :, 1].detach()) + (where[:, :, 3] + 1) * 0.5 * (1 - 1 / where[:, :, 1].detach())
        return torch.stack([1 - px, 1 - py], dim=-1)

    def expand_where(self, where):
        # Takes 4-dimensional vectors, and massages them into 2x3 matrices with elements like so:
        # [s,dx,dy] -> [[s,0,dx],
        #             	[0,s,dy]]

        # where [Nx3]: 		parameters for attention
        assert where.size(1) == 4, '4D-only'
        n = where.size(0)
        out = torch.cat((where.new_zeros((n, 1)), where), dim=1)
        return torch.index_select(out, 1, self.expansion_indices).view(n, 2, 3)

    def image_to_window(self, image, where):
        # image [NxCxHxW]: 	input image
        # where [Nx4]: 		parameters for attention
        assert image.size(0) == where.size(0), 'batch sizes of image and where are not matched'
        n, c, h, w = image.size()
        theta = self.expand_where(self.where_inv(where))
        grid = F.affine_grid(theta, torch.Size((n, c, self.att_size, self.att_size))) #, align_corners=True
        out = F.grid_sample(image.view(n, c, h, w), grid) #, align_corners=True
        out_data = F.grid_sample(image.view(n, c, h, w), grid, mode='nearest') #, mode='nearest', align_corners=True
        #copy only the nearest data and keep gradient from interpolation!
        out.data[:]=out_data.data[:]

        return out

    def window_to_image(self, window, where, image_size, att_s=None):
        assert window.size(2) == window.size(3), 'window should be a square image'
        n, c, window_size, window_size = window.size()
        if att_s==None:
            att_s=self.att_size
        assert window_size == att_s, 'window size is not mathced with attention size'
        theta = self.expand_where(where)
        grid = F.affine_grid(theta, torch.Size((n, c, image_size[0], image_size[1])))# , align_corners=True
        out = F.grid_sample(window.view(n, c, att_s, att_s), grid)  # mode='bilinear', mode='nearest', align_corners=True
        out_data = F.grid_sample(window.view(n, c, att_s, att_s), grid, mode='nearest') # mode='bilinear', mode='nearest', align_corners=True
        #copy only the nearest data and keep gradient from interpolation!
        out.data[:]=out_data.data[:]
        return out

    def forward(self):
        return

    def getInit(self):
        return torch.FloatTensor([2.0, 2.0, 0.0, 0.0])

class SpatialTransformer4(nn.Module):
    #TODO gradient noise: h = v.register_hook(lambda grad: grad * 2)
    def __init__(self, outDim, inpDim, maxZoom=3):
        super(SpatialTransformer4, self).__init__()
        self.Xs_wnd_, self.Ys_wnd_ = torch.meshgrid([torch.linspace(0, 1, outDim), torch.linspace(0, 1, outDim)])
        self.Xs_inp_, self.Ys_inp_ = torch.meshgrid([torch.linspace(0, 1, inpDim), torch.linspace(0, 1, inpDim)])
        self.register_buffer('Xs_wnd', self.Xs_wnd_)
        self.register_buffer('Ys_wnd', self.Ys_wnd_)
        self.register_buffer('Xs_inp', self.Xs_inp_)
        self.register_buffer('Ys_inp', self.Ys_inp_)
        self.att_size = outDim
        self.rec_size = inpDim
        self.ratio = 1 * outDim / inpDim #
        self.maxZoom=maxZoom

    # calculate distortion (vector field from target to window for one axis

    def distortFuncTargetToWdn(self, inp, center, scale, slope=50):
        distA = center
        distB = 1 - center
        pointA = center - distA * 0.8 * scale
        pointB = center + distB * 0.8 * scale
        pointA_out = center - distA * 0.8
        pointB_out = center + distB * 0.8
        scaleA = 1 / pointA
        scaleB = 1 / (1 - pointB)
        sigA = torch.erf(slope * (pointA_out[:, None, None] - inp[None, :, :])) * 0.5 + 0.5
        sigB = torch.erf(slope * (inp[None, :, :] - pointB_out[:, None, None])) * 0.5 + 0.5
        gA = inp[None, :, :] / pointA_out[:, None, None] * (pointA[:, None, None])
        gB = (1 - pointB[:, None, None]) * (
                    (inp[None, :, :] - pointB_out[:, None, None]) / (1 - pointB_out[:, None, None])) + pointB[:, None,
                                                                                                       None]
        gC = (inp[None, :, :] - center[:, None, None]) * (scale[:, None, None]) + center[:, None, None]

        # slope is 0.25, so deriviative (1/ratio) means 4*
        gC2 = 1 / (1 + torch.exp(-(inp[None, :, :] - center[:, None, None]) * (4 / scale[:, None, None]))) + center[:,
                                                                                                             None,
                                                                                                             None] - 0.5

        sig = sigA * gA + sigB * gB + (1 - sigB - sigA) * gC
        return [gA, gB, gC, sigA, sigB, sig, gC2]

    def distortFuncTargetToInp(self, inp, center, scale, slope=50):
        center = center
        inp = inp[None, :, :]
        distA = center
        distB = 1 - center
        pointA = center - distA * 0.8 * scale
        pointB = center + distB * 0.8 * scale
        pointA_out = center - distA * 0.8
        pointB_out = center + distB * 0.8
        scaleA = 1 / pointA
        scaleB = 1 / (1 - pointB)
        pointA = pointA[:, None, None]
        pointB = pointB[:, None, None]
        pointA_out = pointA_out[:, None, None]
        pointB_out = pointB_out[:, None, None]
        center = center[:, None, None]
        scale = scale[:, None, None]
        sigA = torch.erf(slope * (pointA - inp)) * 0.5 + 0.5
        sigB = torch.erf(slope * (inp - pointB)) * 0.5 + 0.5
        gA = inp / pointA * (pointA_out)
        gB = (1 - pointB_out) * ((inp - pointB) / (1 - pointB)) + pointB_out
        gC = (inp - center) * (1 / scale) + center
        # slope is 0.25, so deriviative (1/ratio) means 4*
        gC2 = 1 / (1 + torch.exp(-(inp - center) * (4 / scale))) + center - 0.5
        sig = sigA * gA + sigB * gB + (1 - sigB - sigA) * gC
        return [gA, gB, gC, sigA, sigB, sig, gC2]

    # parameterization: TODO:!=[sx, sy, dx,dy]
    def image_to_window(self, image, where):
        where = torch.tanh(where) * 0.49 + 0.5
        # print("image to wnd:", where)
        # sclae = self.ratio/(1..)
        #consider max zoom factor!
        scalefac = self.ratio / (1 + (where[:, 0]*(self.maxZoom-1)))
        [sigA, sigB, sigC, c1, c2, sig1, tt] = self.distortFuncTargetToWdn(self.Xs_wnd, where[:, 2], scalefac)
        [sigA, sigB, sigC, c3, c4, sig2, tt] = self.distortFuncTargetToWdn(self.Ys_wnd, where[:, 1], scalefac)
        X_hat_re = sig1
        Y_hat_re = sig2
        grid = torch.stack([Y_hat_re * 2 - 1, X_hat_re * 2 - 1], dim=-1)
        wnd = torch.nn.functional.grid_sample(image, grid) # , align_corners=True
        return wnd

    def window_to_image(self, window, where, image_size):
        where = torch.tanh(where) * 0.49 + 0.5
        scalefac = self.ratio / (1 + (where[:, 0]*(self.maxZoom-1)))
        [sigA, sigB, sigC, c1, c2, sig1, tt] = self.distortFuncTargetToInp(self.Xs_inp, where[:, 2], scalefac)
        [sigA, sigB, sigC, c3, c4, sig2, tt] = self.distortFuncTargetToInp(self.Ys_inp, where[:, 1], scalefac)
        # print("wnd to img:", where)
        X_hat_re = sig1
        Y_hat_re = sig2
        grid = torch.stack([Y_hat_re * 2 - 1, X_hat_re * 2 - 1], dim=-1)
        inp_rec = torch.nn.functional.grid_sample(window, grid) #, align_corners=True
        return inp_rec

    def forward(self):
        return

    def getInit(self):
        return torch.FloatTensor([0.5, 0.0, 0.0])  # half size centered

    def where_to_center(self, where):
        # return the center of the window normalized to 0..1
        # px = torch.tanh(where[:,:,1])*0.49+0.5 #torch.tanh(where)*0.49+0.5
        # py = torch.tanh(where[:,:,2])*0.49+0.5
        scale = torch.tanh(where[:, :, 0]) * 0.49 + 0.5
        centerx = torch.tanh(where[:, :, 1]) * 0.49 + 0.5
        centery = torch.tanh(where[:, :, 2]) * 0.49 + 0.5
        scalefac = 0.8 / (1 + (scale*(self.maxZoom-1)))
        px = centerx * (1 - 0.8 * scalefac) + (0.8 * scalefac / 2)
        py = centery * (1 - 0.8 * scalefac) + (0.8 * scalefac / 2)

        return torch.stack([px, py], dim=-1)

    def where_on_image(self, where, image):

        scale = torch.tanh(where[:, :, 0]) * 0.49 + 0.5
        centerx = torch.tanh(where[:, :, 2]) * 0.49 + 0.5
        centery = torch.tanh(where[:, :, 1]) * 0.49 + 0.5

        scalefac = 0.8 / (1 + (scale*(self.maxZoom-1)))
        scale = 0.8

        pA_x = centerx - (centerx) * scale * scalefac
        pB_x = centerx + (1 - centerx) * scale * scalefac

        pA_y = centery - (centery) * scale * scalefac
        pB_y = centery + (1 - centery) * scale * scalefac

        px = centerx * (1 - scale * scalefac) + (scale * scalefac / 2)
        py = centery * (1 - scale * scalefac) + (scale * scalefac / 2)

        lx = image.size()[3] - 1
        ly = image.size()[4] - 1
        xA_ = (torch.round(pA_x * lx))
        xB_ = (torch.round(pB_x * lx))
        yA_ = (torch.round(pA_y * ly))
        yB_ = (torch.round(pB_y * ly))

        for i in range(where.size()[0]):
            for j in range(where.size()[1]):
                xA = int(xA_[i, j])
                xB = int(xB_[i, j])
                yA = int(yA_[i, j])
                yB = int(yB_[i, j])
                if image.size()[2] == 3:
                    image[i, j, :, xA, yA:yB + 1] = torch.tensor([1, 0, 0])[:, None]
                    image[i, j, :, xB, yA:yB + 1] = torch.tensor([1, 0, 0])[:, None]
                    image[i, j, :, xA:xB + 1, yA] = torch.tensor([1, 0, 0])[:, None]
                    image[i, j, :, xA:xB + 1, yB] = torch.tensor([1, 0, 0])[:, None]
                else:
                    image[i, j, :, xA, yA:yB + 1] = torch.tensor([1])[:, None]
                    image[i, j, :, xB, yA:yB + 1] = torch.tensor([1])[:, None]
                    image[i, j, :, xA:xB + 1, yA] = torch.tensor([1])[:, None]
                    image[i, j, :, xA:xB + 1, yB] = torch.tensor([1])[:, None]

class Model_vision(nn.Module):
    def __init__(self, seed, num_context_frames, intention_size, vision_args, motor_args, integration_args, language_args, attention_args,
                 do_global_spatial_transformer, do_center_loss):
        super(Model_vision, self).__init__()
        torch.manual_seed(seed)
        self.num_context_frames = num_context_frames
        self.intention_size = intention_size
        self.is_lang = language_args["is_lang"]
        self.is_pb = language_args['is_pb']
        self.is_lateral = motor_args['is_lateral']
        self.is_bottomup = motor_args['is_bottomup']
        self.L0Memory = False
        self.useL0Memory_v9 = False

        self.useL0Memory_feedback_mem_to_lstm = True
        self.L0MemoryL1Reg = -1
        self.L0Memory_trainsignal = vision_args['central_vision']['layers'][0].get('memory_trainsignal', False)
        self.canvas_delay_time = vision_args['canvas_delay']
        self.transformer_slope = vision_args['transformer_slope']
        self.L0Memory_transformer = None
        self.L0Memory_transformer_dims = 0
        self.attention_dim = attention_args['dim']
        self.cloop_ratio = vision_args['cloop_ratio']
        # collect parameters:
        v_dim = vision_args['dim']
        v_num_channels = vision_args['num_channels']

        cv_dim = vision_args['central_vision']['dim']
        cv_layers_spec = vision_args['central_vision']['layers']
        self.hasLowLevelMemory = vision_args.get('low_level_memory', True)
        self.vis_dropout_rate = vision_args.get('dropout_rate')
        self.num_joints = motor_args['num_joints']
        self.joint_enc_dim = motor_args['joint_enc_dim']
        self.joint_is_softmax = motor_args.get('is_softmax', True)
        m_dim = self.num_joints * self.joint_enc_dim
        m_layers_spec = motor_args['layers']
        # self.is_pvrnn = integration_args['is_pvrnn']
        i_layers_spec = integration_args['layers']
        l_layers_spec = language_args['layers']
        self.w1 = []
        self.w = []
        for l in range(len(integration_args['layers'])):
            self.w1.append(integration_args['layers'][l]['w1'])
            self.w.append(integration_args['layers'][l]['w'])
        v_mem_dim = v_dim

        if do_global_spatial_transformer:
            # TODO
            transformer_maxZoom = vision_args.get('spatial_transformer_maxZoom', 2.0)
            self.transformer = SpatialTransformer4(outDim=cv_dim, inpDim=v_dim, maxZoom=transformer_maxZoom)
        else:
            self.transformer = SpatialTransformer(outDim=cv_dim, inpDim=v_dim)
        self.add_module("transformer", self.transformer)

        # it is assumed that the number of layers of motor, central vision and peripheral vision are equal,
        # due to lateral connections!
        assert len(cv_layers_spec) == len(
            m_layers_spec), 'vision and motor modalities are not equal in number of layers'
        integration_hid_size = i_layers_spec[0]['hid_size'] # size of the layer the combines vision and motor

        # calculate missing parameters from aready given ones:
        cv_prev_filter_size = cv_dim
        for l in range(len(m_layers_spec)):
            cv_layers_spec[l]['dim_lateral'] = m_layers_spec[l]['hid_size']
            # cv_layers_spec[l]['dim_topdown'] = 0
            if l == (len(m_layers_spec) - 1):
                cv_layers_spec[l]['dim_lateral'] += integration_hid_size
            if l > 0:
                cv_prev_filter_size = cv_layers_spec[l - 1]['filter_size']
            else:
                if cv_layers_spec[l].get('memory') == False:
                    # activate L0 memory
                    self.useL0Memory = False
                else:
                    self.useL0Memory = True
                    self.useL0Memory_feedback_mem_to_lstm=cv_layers_spec[l].get('memory_feedback_lstm', True)
                    print("Model uses L0Memory architecture!")
                    # if cv_layers_spec[l].get('memory_version', 1) == 9:
                    #     self.useL0Memory_v9 = True
                    #     print("Model uses direct Dynamic Memory v9")
                    self.L0MemoryL1Reg = cv_layers_spec[l].get('memory_L1reg', -1.0)
                    if cv_layers_spec[l].get('memory_transformer', False):
                        # activate L0Memory transformer
                        # if self.useL0Memory_v9 == True:
                        #     #x-y scaling allowed:
                        #     self.L0Memory_transformer_dims = 4
                        #     #upscaled representation:
                        #     self.L0Memory_transformer = SpatialTransformer_2Dscaling(outDim=cv_dim*4, inpDim=cv_dim*4)
                        #     self.L0Memory_transformer_cv_dim = cv_dim
                        #     self.L0Memory_transformer_dim = cv_dim*4
                        #     if do_global_spatial_transformer:
                        #         # TODO
                        #         self.transformer_upscale = SpatialTransformer4(outDim=cv_dim*4, inpDim=v_dim, maxZoom=transformer_maxZoom)
                        #         self.transformer_upscale.ratio=self.transformer.ratio
                        #     else:
                        #         self.transformer_upscale = SpatialTransformer(outDim=cv_dim*4, inpDim=v_dim)
                        #     self.add_module("transformer_upscale", self.transformer_upscale)
                        # else:
                        self.L0Memory_transformer = SpatialTransformer(outDim=cv_dim, inpDim=cv_dim)
                        self.L0Memory_transformer_dims = 3
                        self.add_module("L0Memory_transformer", self.L0Memory_transformer)
            cv_layers_spec[l]['dim_bottomup'] = cv_prev_filter_size
            cv_layers_spec[l]['filter_size'] = round(cv_prev_filter_size / cv_layers_spec[l]['downscale_factor'])
            cv_prev_filter_size = cv_layers_spec[l]['filter_size']
            if l > 0:
                cv_layers_spec[l - 1]['dim_topdown'] = cv_layers_spec[l]['filter_size']

        # added extra channels
        # two extra outputs: mixing memory update and output mix masks
        # input * 2 : input + memory input
        l0MemoryInputs = 0
        l0MemoryOutputs = 0
        if self.useL0Memory:
            if self.useL0Memory_feedback_mem_to_lstm:
                l0MemoryInputs = v_num_channels  # RGB input of L0 memory
            l0MemoryOutputs = 2  # two extra outputs: mixing memory update and output mix masks for L0Memory!

            # two extra outputs: mixing memory update and output mix masks, only one stream so no stream mixing dimension
        self.cv_net = VisionStream(dropout_rate=self.vis_dropout_rate, num_inp_channels=v_num_channels * 2 + l0MemoryInputs,
                                   num_out_channels=v_num_channels + 2 + l0MemoryOutputs,
                                   layers_spec=cv_layers_spec, is_bottomup=self.is_bottomup, is_lateral=self.is_lateral)
        # TODO
        self.add_module("central_vision", self.cv_net)          # vision module for old results, still not sure why we have two copies of cv_net
        #
        m_cv_in_sizes = []
        for l in range(len(m_layers_spec)):
            cv_input_dim = 0
            if l == 0:
                # add further input dimensions, external input in lowerst level:
                cv_input_dim += m_dim + attention_args['dim']
            # topdown
            if l < (len(m_layers_spec) - 1):
                cv_input_dim += m_layers_spec[l + 1]['hid_size']
            elif len(m_layers_spec) == 1:
                #only one layer in motor and vision
                cv_input_dim += integration_hid_size
            else:
                # uppest layer: intention input
                cv_input_dim += integration_hid_size
            if self.is_lateral:
                cv_input_dim += (cv_layers_spec[l]['filter_size'] ** 2) * cv_layers_spec[l]['num_filter']
            if self.is_bottomup and l!=0:
                # bottom up
                cv_input_dim += m_layers_spec[l - 1]['hid_size']
            m_cv_in_sizes.append(cv_input_dim)
        self.m_cv_net = Motor_vision(in_sizes=m_cv_in_sizes, **motor_args)          # motor or proprioception module

        integration_motor_input = m_layers_spec[-1]['hid_size']
        integration_vision_input = (cv_layers_spec[-1]['filter_size'] ** 2) * cv_layers_spec[-1]['num_filter']

        # layer order is reversed for pvrnn so top layer is the associative layer that combines vision abd notor
        i_in_sizes = []
        # if len(i_layers_spec) == 1 and self.is_pvrnn:
        #     i_input_dim = 0
        #     i_in_sizes.append(i_input_dim)
        # ToDo: add a hyperparameter to specify the presence of bottom up connections only in pvrnn
        # if self.is_pvrnn:
        for l in range(len(i_layers_spec)):

            i_input_dim = 0
            if l == len(i_layers_spec) - 1:
                i_input_dim += integration_vision_input + integration_motor_input
            else:
                i_input_dim += 0
            # topdown
            if l < (len(i_layers_spec) - 1):
                i_input_dim += i_layers_spec[l + 1]['hid_size']
            i_in_sizes.append(i_input_dim)
        #
        # if self.is_pvrnn:
        #     for l in range(len(i_layers_spec)):
        #         i_input_dim = 0
        #         # print(i_input_dim)
        #         if l == 0 and self.is_bottomup:
        #             # add further input dimensions, external input in lowest level:
        #             i_input_dim += integration_motor_input + integration_vision_input
        #         else:
        #             # if integration has bottom up i_input_dim += i_layers_spec[l - 1]['hid_size']
        #             i_input_dim += 0  #i_layers_spec[l]['hid_size'] # no bottom up
        #         # topdown
        #         if l < (len(i_layers_spec) - 1):
        #             i_input_dim += i_layers_spec[l + 1]['hid_size']
        #         i_in_sizes.append(i_input_dim)
        # lstm only integration network
        else:
            for l in range(len(i_layers_spec)):
                i_input_dim = 0
                # print(i_input_dim)
                if self.is_bottomup:
                    if l == 0:
                        i_input_dim += integration_motor_input + integration_vision_input
                    else:
                        i_input_dim += i_layers_spec[l - 1]['hid_size']
                if l < (len(i_layers_spec) - 1):
                    # topdown
                    i_input_dim += i_layers_spec[l + 1]['hid_size']
                i_in_sizes.append(i_input_dim)
        # print("i_insize={}".format(i_in_sizes))
        self.integration_net = Integration_bind(in_sizes=i_in_sizes, **integration_args)  # this is defining the integration module

        if self.is_lang:
            # if self.is_pb:
            print("language PB binding")
                # if self.is_integpb:
                #     self.fc_pb_to_integ = nn.Sequential(
                #         nn.Linear(l_layers_spec[0]['pb_size'], i_layers_spec[0]['hid_size']))  # integ pb to d
                # else:
            self.fc_integ_lang = nn.Sequential(
                nn.Linear(i_layers_spec[0]['hid_size'], l_layers_spec[0]['pb_size']))  # parametric bias binding #.requires_grad_(False)
                    # self.fc_integ_lang = nn.Sequential(
                    #     nn.Linear(i_layers_spec[0]['z_size'], l_layers_spec[0]['pb_size']))
            # else:
            #     print("language initial state binding")
            #     self.fc_integ_lang = nn.Sequential(nn.Linear(i_layers_spec[0]['hid_size'], l_layers_spec[0]['hid_size'])) # intial state binding

            lang_dim = language_args['dim']
            l_in_sizes = []
            for l in range(len(l_layers_spec)):
                l_input_dim = 0 #l_layers_spec[l]['hid_size']
                if l == 0:
                    # add further input dimensions, external input in lowest level:
                    l_input_dim += lang_dim
                else:
                    # bottom up
                    l_input_dim += l_layers_spec[l - 1]['hid_size']
                # topdown
                if l < (len(l_layers_spec) - 1):
                    l_input_dim += l_layers_spec[l + 1]['hid_size']

                l_in_sizes.append(l_input_dim)
            self.language_net = Language(l_in_sizes, **language_args)  # language lstm module

            self.fc_l_pred = nn.Sequential(nn.Linear(l_layers_spec[0]['hid_size'], lang_dim))

        self.m_enc_dim = m_dim
        # dimensionality of states of lowest layer
        motor_dims = m_layers_spec[0]['hid_size']
        if m_dim > 0:
            self.fc_m_pred = nn.Sequential(
                nn.Linear(motor_dims, motor_args['readout_hid_size']),
                nn.LayerNorm(motor_args['readout_hid_size']),
                nn.ReLU(inplace=True),
                nn.Linear(motor_args['readout_hid_size'], m_dim)
            )
        else:
            self.fc_m_pred = None

        self.pb_dims = l_layers_spec[0]['pb_size']

        self.fc_att_where = nn.Sequential(
            nn.Linear(motor_dims, attention_args['readout_hid_size']),
            nn.LayerNorm(attention_args['readout_hid_size']),
            nn.ReLU(inplace=True),
            nn.Linear(attention_args['readout_hid_size'], attention_args['dim'] + self.L0Memory_transformer_dims)
        )

        self.fc_att_where[-1].weight.data.fill_(0)

        transformer_init = vision_args.get('spatial_transformer_init', None)
        if transformer_init is None:
            self.fc_att_where[-1].bias.data[0:attention_args['dim']] = self.transformer.getInit()
        else:
            self.fc_att_where[-1].bias.data[0:attention_args['dim']] = torch.FloatTensor(
                transformer_init)  # half size centered

        if self.L0Memory_transformer is not None:
            # if self.useL0Memory_v9:
            #     self.fc_att_where[-1].bias.data[attention_args['dim']:] = torch.FloatTensor(
            #         [0.0, 0.0, 0.0, 0.0])  # 1:1 mapping - no transformation
            # else:
            self.fc_att_where[-1].bias.data[attention_args['dim']:] = torch.FloatTensor(
                [1.0, 0.0, 0.0])  # 1:1 mapping - no transformation

        # store configurations for later use:
        self.config = type("modelconfig", (object,), {})   #empty dictionary object??
        self.config.v_dim = v_dim  # size of visual stream
        self.config.v_mem_dim = v_mem_dim  # size of visual memory
        self.config.v_num_channels = v_num_channels  # channels for visual stream
        self.config.v_stream_dims = [cv_dim]  # size of splitted visual streams | pv_dim,
        self.config.v_streams_spec = [cv_layers_spec]  # layer properties | pv_layers_spec,
        self.config.m_streams_spec = m_layers_spec  # motor stream configuration
        self.config.l_streams_spec = l_layers_spec
        self.config.integration_spec = i_layers_spec
        self.att_size = cv_dim  # sample size for attention window

        # count number of parameters:
        cv_layer_num_parameters = []
        m_layer_num_parameters = []
        l_layer_num_parameters = []
        i_layer_num_parameters = []
        # currently: vision and motor streams have same number of layers
        for l in range(len(self.config.m_streams_spec)):
            cv_layer_num_parameters.append(
                (self.config.v_streams_spec[0][l]['filter_size'] ** 2) * self.config.v_streams_spec[0][l]['num_filter'])
            m_layer_num_parameters.append(self.config.m_streams_spec[l]['hid_size'])

        for l in range(len(self.config.integration_spec)):
            i_layer_num_parameters.append(self.config.integration_spec[l]['hid_size'])

        for l in range(len(self.config.l_streams_spec)):
            l_layer_num_parameters.append(self.config.l_streams_spec[l]['hid_size'])

        # save for later use:
        self.config.cv_layer_num_parameters = cv_layer_num_parameters
        self.config.m_layer_num_parameters = m_layer_num_parameters
        self.config.i_layer_num_parameters = i_layer_num_parameters
        self.config.l_layer_num_parameters = l_layer_num_parameters
    def split_init_states(self, hidden, cell):
        """
        splitting the initial states vector to get initial states of each layer
        for when pvrnn is not used
        """

        language_states = []
        cv_states = []
        m_cv_states = []
        dim_cv = sum(self.config.cv_layer_num_parameters)
        dim_motor = sum(self.config.m_layer_num_parameters)
        dim_integration = sum(self.config.i_layer_num_parameters)
        if self.is_lang:
            dim_language = sum(self.config.l_layer_num_parameters)
            cv_init_hidden, m_cv_init_hidden, integration_init_hidden, language_init_hidden= hidden.split(
                [dim_cv, dim_motor, dim_integration, dim_language], dim=1)
            cv_init_cell, m_cv_init_cell, integration_init_cell, language_init_cell = cell.split(
                [dim_cv, dim_motor, dim_integration, dim_language], dim=1)
            l_start = 0
            for l in range(len(self.config.l_streams_spec)):
                l_end = l_start + self.config.l_streams_spec[l]['hid_size']
                language_states.append(
                    (language_init_hidden[:, l_start:l_end], language_init_cell[:, l_start:l_end])
                )
                l_start = l_end
        else:
            cv_init_hidden, m_cv_init_hidden, integration_init_hidden = hidden.split(
                [dim_cv, dim_motor, dim_integration], dim=1)
            cv_init_cell, m_cv_init_cell, integration_init_cell = cell.split(
                [dim_cv, dim_motor, dim_integration], dim=1)

        # vision for all layers:
        cv_start = 0
        m_cv_start = 0

        for l in range(len(self.config.m_streams_spec)):

            cv_size = self.config.v_streams_spec[0][l]['filter_size']
            cv_num = self.config.v_streams_spec[0][l]['num_filter']
            cv_total = (cv_size ** 2) * cv_num
            cv_end = cv_start + cv_total
            cv_states.append(
                (cv_init_hidden[:, cv_start:cv_end].view(cv_init_hidden.size(0), cv_num, cv_size, cv_size),
                 cv_init_cell[:, cv_start:cv_end].view(cv_init_cell.size(0), cv_num, cv_size, cv_size))
            )

            m_cv_end = m_cv_start + self.config.m_streams_spec[l]['hid_size']
            m_cv_states.append(
                (m_cv_init_hidden[:, m_cv_start:m_cv_end], m_cv_init_cell[:, m_cv_start:m_cv_end])
            )
            cv_start = cv_end
            m_cv_start = m_cv_end
        # integration:
        integration_states = []
        i_start = 0
        for l in range(len(self.config.integration_spec)):
            i_end = i_start + self.config.integration_spec[l]['hid_size']
            integration_states.append(
                (integration_init_hidden[:, i_start:i_end], integration_init_cell[:, i_start:i_end])
            )
            i_start = i_end
        return integration_states, cv_states, m_cv_states, language_states

    def split_A_terms(self, A_mu, A_logsigma):
        A_mus, A_logsigmas = [], []
        i_start = 0
        if len(A_mu.shape) == 2:
            for l in range(len(self.config.integration_spec)):
                i_end = i_start + self.config.integration_spec[l]['z_size']
                A_mus.append(A_mu[:, i_start:i_end])
                A_logsigmas.append(A_logsigma[:, i_start:i_end])
                i_start = i_end
        elif len(A_mu.shape) == 1:
            for l in range(len(self.config.integration_spec)):
                i_end = i_start + self.config.integration_spec[l]['z_size']
                A_mus.append(A_mu[i_start:i_end])
                A_logsigmas.append(A_logsigma[i_start:i_end])
                i_start = i_end
        else:
            for l in range(len(self.config.integration_spec)):
                i_end = i_start + self.config.integration_spec[l]['z_size']
                A_mus.append(A_mu[:, :, i_start:i_end])
                A_logsigmas.append(A_logsigma[:, :,  i_start:i_end])
                i_start = i_end
        return A_mus, A_logsigmas

    def variable_hook2(self, grad):
        #print('hook2', torch.mean(grad))
        return grad

    def variable_hook(self, grad):
        #print('variable hook')

        N = len(self.gradcollection)
        N_r = self.v_mask_memup_states_wnd.shape[1]-N
        mask = self.v_mask_memup_states_wnd[:, N,:,:]
        #print('variable hook len: '+ str(len(self.gradcollection)) + ' ' + str(self.v_mask_memup_states_wnd.shape) + ' res ' + str(N_r))
        #print('grad', torch.max(grad.abs()))
        res = grad
        #if N_r>0:
        val_div = torch.clamp(mask*N_r, min=1)
        res = res / torch.clamp(mask.data*N_r, min=1)
        #print('gradmaxmin', torch.max(val_div), ' ', torch.min(val_div), ' ', N_r)
        self.gradcollection.append(val_div[0, :, :, :])

        #print('hook1', torch.mean(res))
        return res #grad / (mask*N_r)  #*0.5 destroy memory!!

    def create_init_states(self):
        """
        create initial states for each layer of LSTM
        """
        dim_cv = sum(self.config.cv_layer_num_parameters)
        dim_motor = sum(self.config.m_layer_num_parameters)
        dim_integration = sum(self.config.i_layer_num_parameters)
        if self.is_lang:
            dim_language = sum(self.config.l_layer_num_parameters)
        if self.is_lang:
            num_parameters = (dim_integration + dim_cv + dim_motor + dim_language) #+ dim_pv
        else:
            num_parameters = (dim_integration  + dim_cv + dim_motor) #+ dim_pv
        parameter = torch.zeros(num_parameters)
        return parameter
    def pvrnn_init_states(self, layers, seq_len=0):
        """
        get pvrnn A parameterssss
        need to set up for multiple layers
        """
        # for l in range(len(layers)):
        #     A_mu = nn.Parameter(torch.randn(seq_len. layers[l]['z_size']))
        #     A_logsigma = nn.Parameter(torch.randn(seq_len, layers[l]['z_size']))
        z_size = sum([layers[l]['z_size'] for l in range(len(layers))])
        A_mu = nn.Parameter(torch.randn(seq_len, z_size))
        A_logsigma = nn.Parameter(torch.randn(seq_len, z_size))

        return A_mu, A_logsigma

    def forward(self, vision, motor, language, cell_mu, cell_logvar, pv_mu, pv_logvar, pv_mup_i, pv_logvar_i,
                lang_pb=None, gen_lang=False):
        """
        forward model of the network
        """

        n = vision.size(0)
        # print("n={}".format(n))
        device = vision.device
        L0MemoryReg = []
        cell = cell_mu + torch.exp(0.5 * cell_logvar) * cell_logvar.new(cell_logvar.size()).normal_()
        hidden = F.tanh(cell)
        # convert from 'batch' (N) first to 'seq_len' (T) first

        vision = vision.transpose(0, 1)
        if len(motor) > 0: motor = motor.transpose(0, 1)
        lang_init_preds = []
        lang_pb_preds = []
        integration_states, cv_states, m_cv_states, lang_states = self.split_init_states(cell, hidden)
        pv_mu, pv_logvar = self.split_A_terms(pv_mu, pv_logvar)
        pv_mup_i, pv_logvar_i = self.split_A_terms(pv_mup_i, pv_logvar_i)
        if self.is_lang:
        #     if lang_init==None:
            lang_init = torch.zeros(language.shape[0], self.config.l_layer_num_parameters[0]).data
            lang_states = [lang_init.to(cell.device), torch.zeros_like(lang_init).to(cell.device)]
        integration_state_gates, cv_state_gates, m_cv_state_gates = [], [], []
        cv_states_ = []
        m_cv_states_ = []
        raw_cv_pred = []
        # if self.is_pvrnn:
        #replace lstm intitial states with zeros when using pvrnn
        for i in range(len(cv_states)):
            cv_states_.append([torch.zeros_like(cv_states[i][0].data), torch
                              .zeros_like(cv_states[i][1].data)])
        for i in range(len(m_cv_states)):
            m_cv_states_.append([torch.zeros_like(m_cv_states[i][0].data), torch.zeros_like(m_cv_states[i][1].data)])
        cv_states, m_cv_states, = cv_states_, m_cv_states_
        for i in range(len(integration_states)):
            integration_states[i] = [torch.zeros_like(integration_states[i][0].data), torch.zeros_like(integration_states[i][0].data)]


        #intial states

        m_cv_states = [m_cv_states]
        integration_states = [integration_states]
        v_states = [[cv_states]]

        v_mask_mix_states = []
        v_mask_memout_states = []
        v_mask_memup_states = []
        self.v_mask_memup_states_wnd=[]
        canvas = None       # buffer from VWM-1 to VWM-2
        L0Memory = []
        cv_predictions = []
        # cv_l0_memorystates = []
        # cv_l1_memorystates = []
        v_predictions = []
        m_predictions = []
        l_predictions = []
        memory_states = []
        # memory states of VWM-2
        L0Memory_states = []
        L0Memory_states_transformed = []
        L0Memory_states_outmix = []
        L0Memory_states_feedback = []
        L0Memory_updates = []
        L0MemoryFiltered = []
        L0_memselection = []
        L0_memmaskupscale = []
        canvas_ = []
        l0memupdatemask_pred = []
        memory_feedback = []

        attention_wheres = []
        l0mem_updatemask = []
        integration_top_states = []
        mu_p_list, sigma_p_list = [], []
        mu_q_list, sigma_q_list = [], []
        z_list = []
        pv_kl = 0.

        self.gradcollection=[]
        for step, (ext_v, ext_m) in enumerate(itertools.zip_longest(vision, motor)):
            # print("step = {}".format(step))
            # e.g. first input is initial input, the following are predictions (test) or a mixture (training) of train targets and predictions:
            if step > self.num_context_frames - 1:
                if self.training: # ToDo: try adding noise to bottom up input
                    feedback_ratio = self.cloop_ratio          # closed loop ratio, make this an argument in config
                    v = (1. - feedback_ratio) * ext_v + feedback_ratio * v_predictions[-1]
                    if ext_m is not None: m = (1. - feedback_ratio) * ext_m + feedback_ratio * torch.exp(
                        m_predictions[-1])
                else:
                    if gen_lang:
                        feedback_ratio = self.cloop_ratio
                    else:
                        feedback_ratio = self.cloop_ratio
                    v = (1. - feedback_ratio) * ext_v + feedback_ratio * v_predictions[-1]
                    if ext_m is not None: m = (1. - feedback_ratio) * ext_m + feedback_ratio * torch.exp(
                        m_predictions[-1])
            else:
                v = ext_v #torch.zeros_like(ext_v)
                m = ext_m #torch.zeros_like(ext_m)

            # current time step hidden states of different network modules
            integration_current_state = []
            for l in range(len(self.config.integration_spec)):
                assert (len(integration_states[-1]) == len(self.config.integration_spec))
                integration_current_state.append(integration_states[-1][l])
            cv_current_state = []
            m_cv_current_state = []

            for l in range(len(self.config.m_streams_spec)):
                assert (len(v_states[-1][0]) == len(self.config.m_streams_spec))
                cv_current_state.append(v_states[-1][0][l])
                m_cv_current_state.append(m_cv_states[-1][l])
            if step == 0:
                read_where = self.fc_att_where(m_cv_current_state[0][0]) #+ self.fc_langpb_to_att(torch.zeros_like(lang_pb).to(lang_pb.device))
                attention_wheres.append(read_where)

            m_cv_incomings = []            # list of inputs to all layers of motor module
            for l in range(len(self.config.m_streams_spec)):
                if self.is_lateral:
                    if l == 0 and len(self.config.m_streams_spec) == 1:
                        m_cv_incomings.append(
                            (cv_current_state[l][0].view(n, -1),
                             attention_wheres[-1][:, :self.attention_dim], integration_current_state[0][0],))
                    elif l == 0:
                        m_cv_incomings.append(
                            (cv_current_state[l][0].view(n, -1),
                             attention_wheres[-1][:, :self.attention_dim],))
                    elif l < (len(self.config.m_streams_spec) - 1):
                        m_cv_incomings.append((cv_current_state[l][0].view(n, -1),))
                    else:
                        m_cv_incomings.append((
                            cv_current_state[l][0].data.view(n, -1), integration_current_state[0][0],))
                else:
                    if l == 0 and len(self.config.m_streams_spec) == 1:
                        m_cv_incomings.append(
                            (attention_wheres[-1][:, :self.attention_dim], integration_current_state[0][0]))
                    elif l == 0:
                        m_cv_incomings.append(
                            (attention_wheres[-1][:, :self.attention_dim],))
                    elif l < (len(self.config.m_streams_spec) - 1):
                        m_cv_incomings.append(torch.Tensor([]))
                    else:
                        m_cv_incomings.append(
                            (integration_current_state[0][0],))
            # get next time step states and output from motor module
            m_cv_next_state, m_cv_gates = self.m_cv_net(m, m_cv_incomings, m_cv_current_state)   ## equation 8
            m_cv_state_gates.append(m_cv_gates)
            if self.fc_m_pred is not None:
                #equation 6
                m_pred = self.fc_m_pred(m_cv_next_state[0][0])
                # reshape motor prediction for applying softmax or log softmax to each joint
                m_pred = m_pred.view(-1, self.num_joints, self.joint_enc_dim)
                if self.joint_is_softmax:
                    m_pred = F.log_softmax(m_pred, dim=2)
                m_pred = m_pred.view(-1, self.m_enc_dim)
            else:
                m_pred = None
            #equation 14, 15
            write_where = self.fc_att_where(m_cv_next_state[0][0]) #+ self.fc_langpb_to_att(lang_pb)

            attention_wheres.append(write_where)
            # split vision:
            if step < self.num_context_frames:
                # fill canvas with given data for num_context_frames.
                if step == 0:
                    prev_canvas_filtered = ext_v
                else:
                    prev_canvas_filtered = canvas
                canvas = ext_v
            v_merged = torch.cat([v, prev_canvas_filtered], dim=1)

            cv_in_current = self.transformer.image_to_window(v_merged, attention_wheres[-2][:, :self.attention_dim]) #equation 5
            # add L0 memory to cv_in_current:
            if self.useL0Memory:
                if step < self.num_context_frames:
                    if step == 0:
                        # init memory with transformed input vision in first step
                        L0Memory = cv_in_current[:, 0:self.config.v_num_channels].detach()
                        # if self.useL0Memory_v9 == True:
                        #     L0Memory=F.interpolate(L0Memory.data*0, size=self.L0Memory_transformer_dim) #cv_dim*4
                    # 1:1 memory content is used as feedback information during initial context frames
                    if self.useL0Memory_feedback_mem_to_lstm:
                        L0MemoryFiltered = L0Memory
                else:
                    0
                # gen feedback from mask of  previous timestep
                # L0MemoryFiltered = L0Memory*(1.-L0MemoryOutMix)
                # generated from last run, later in code!
                # attach memory to input of L0
                if self.useL0Memory_feedback_mem_to_lstm:
                    cv_in_current = torch.cat([cv_in_current, L0MemoryFiltered], dim=1)

            cv_lateral_inp = []
            for l in range(len(self.config.m_streams_spec)):
                if l < (len(self.config.m_streams_spec) - 1):
                    cv_lateral_inp.append((m_cv_current_state[l][0],))
                else:
                    cv_lateral_inp.append((m_cv_current_state[l][0],
                                           integration_current_state[0][0]))  # layer 0, even only one layer till now!
            # output from ConvLSTM
            cv_next_state, cv_pred, cv_gates = self.cv_net(cv_in_current, cv_current_state, cv_lateral_inp) ## equation 7

            cv_state_gates.append(cv_gates)
            vision_startIdx = 2
            cv_pred[:, 0:2] = F.sigmoid(cv_pred[:, 0:2])
            # extract 2x L0Memory control signals!
            #equation 13
            if self.useL0Memory:
                # extract L0 Memory signals (2 dimensions)
                L0MemSignals = 2
                L0MemorySignals = F.sigmoid(cv_pred[:, vision_startIdx:vision_startIdx + L0MemSignals])
                vision_startIdx = vision_startIdx + L0MemSignals
                L0MemoryMask = L0MemorySignals[:, 0:1]
                L0MemoryOutMix = L0MemorySignals[:, 1:2]

            # image predicted by ConvLSTM
            cv_pred[:, vision_startIdx:] = F.tanh(cv_pred[:, vision_startIdx:])

            if self.useL0Memory:
                # apply L0Memory mix to cv vision prediction:
                if self.L0Memory_transformer is None:
                    L0Memory_selection = L0Memory
                else:
                    if (self.useL0Memory_v9 == True):
                        attention_cfg = attention_wheres[-2][:,self.attention_dim:]
                        #limit scaling to 0.5 to 1.5
                        attention_cfg[:,0]=F.sigmoid(attention_cfg[:,0])+0.5
                        attention_cfg[:, 1] = F.sigmoid(attention_cfg[:, 1]) + 0.5
                        #limit movement to +- 0.5
                        attention_cfg[:, 2] = F.tanh(attention_cfg[:, 2])*0.5
                        attention_cfg[:, 3] = F.tanh(attention_cfg[:, 3])*0.5
                        L0Memory_selection = self.L0Memory_transformer.image_to_window(L0Memory, attention_cfg)
                    else:
                        L0Memory_selection = self.L0Memory_transformer.image_to_window(L0Memory, attention_wheres[-1][:,self.attention_dim:])
                    #transform memOutmix as well ? static mask what to overwrite (before transformation?)

                if self.useL0Memory_v9 == True:
                    #Downsample Memory!
                    L0Memory_selection_downsampled = F.interpolate(L0Memory_selection, size=self.L0Memory_transformer_cv_dim)
                    L0MemoryOutMix_I = L0MemoryOutMix.clone()
                    #L0MemoryOutMix.register_hook(self.variable_hook2)
                    #L0MemoryOutMix_I.register_hook(self.variable_hook)
                    cv_pred_img = L0MemoryOutMix_I * cv_pred[:, vision_startIdx:] + (1. - L0MemoryOutMix_I) * L0Memory_selection_downsampled
                    # if self.training:
                    #     cv_pred_img = 0.9*cv_pred_img + 0.1*L0Memory_selection_downsampled # only for training
                else:
                    cv_pred_img = L0MemoryOutMix * cv_pred[:, vision_startIdx:] + (1. - L0MemoryOutMix) * L0Memory_selection

                if step >= self.num_context_frames:
                    # only after presenting context frames
                    if self.useL0Memory_feedback_mem_to_lstm:
                        # if self.useL0Memory_v9 == True:
                        #     print('not supposed to be called! Feedback not used for v8')
                        #     L0MemoryFiltered = (1.0 - L0MemoryOutMix) * L0Memory #deactivated in config!!
                        # else:
                        L0MemoryFiltered = (1.0 - L0MemoryOutMix) * L0Memory_selection
                    ###########
                    if self.useL0Memory_v9 == True:
                        # memory_states collection of canvas
                        l0mem_delay_len = self.canvas_delay_time # 5 steps back in time!
                        l0mem_idx = step - l0mem_delay_len - 1
                        if l0mem_idx < 0:
                            canvas_delay = canvas.data * 0
                        else:
                            canvas_delay = memory_states[l0mem_idx]
                        # canvas_delay[memory_mix_mask<0.5]=0.0
                        # update_signal = (memory_mix_mask>0.5).float() * canvas_delay #deleted content of memory
                        # zero - infinite gradinent ?!
                        # update_mask_L0Mem = cv_pred[:, 0:1]
                        update_mask_L0Mem = self.transformer.window_to_image(cv_pred[:, 0:1], write_where,
                                                         image_size=[self.config.v_dim,
                                                                     self.config.v_dim])
                        update_mask_L0Mem = torch.sigmoid((update_mask_L0Mem - self.transformer_slope) * 50.0)  # decision  to be 0 or one
                        # update_mask_L0Mem = memory_mix_mask
                        update_signal = update_mask_L0Mem * canvas_delay.data  # deleted content of memory
                        l0memupdatemask_pred.append(cv_pred[:, 0:1])
                        # L0Memory_updates.append(update_signal)
                        update_signal = self.transformer_upscale.image_to_window(update_signal,  write_where[:,0:self.attention_dim].detach())
                        L0MemoryMask_upscale = F.interpolate(L0MemoryMask, size=self.L0Memory_transformer_dim) #, mode='nearest' cv_dim*4
                        update_signal_downsampled = F.interpolate(update_signal, size=self.L0Memory_transformer_cv_dim)
                        # for plotting
                        L0Memory_updates.append(update_signal_downsampled)
                        canvas_.append(canvas_delay.data)
                        l0mem_updatemask.append(update_mask_L0Mem)
                        L0_memselection.append(L0Memory_selection)
                        L0_memmaskupscale.append(L0MemoryMask_upscale)
                        # dynamic memory, memory always follows update
                        # VWM-2
                        L0Memory = (1. - L0MemoryMask_upscale) * L0Memory_selection + L0MemoryMask_upscale * update_signal.detach()
                    #
                    else:
                        L0Memory = (1. - L0MemoryMask) * L0Memory + L0MemoryMask * cv_pred_img
                        # for plotting
                        L0Memory_updates.append(torch.zeros_like(L0MemoryMask))
                        L0_memselection.append(L0Memory_selection)
                        L0_memmaskupscale.append(L0MemoryMask)
                        canvas_.append(torch.zeros_like(L0MemoryMask))
                        l0mem_updatemask.append(torch.zeros_like(L0MemoryMask))
                        l0memupdatemask_pred.append(cv_pred[:, 0:1])

                    # calc regularization if necessary:
                    if self.L0MemoryL1Reg > 0:
                        # if self.useL0Memory_v9 == True:
                        #     # L0MemReg = torch.mean(((cv_pred[:, 1:2].detach()*10)+1)*torch.abs(L0MemoryOutMix))
                        #     L0MemReg = torch.mean(torch.abs(L0MemoryOutMix))
                        #     # not working!
                        # else:
                        L0MemReg = torch.mean(torch.abs(L0MemoryOutMix)) + torch.mean(torch.abs(L0MemoryMask))
                        L0MemoryReg.append(self.L0MemoryL1Reg * L0MemReg)
            else:
                cv_pred_img = cv_pred[:, vision_startIdx:]

            #ADD hook to check gradient!!
            #accumulated gradient of memory loop ?
            #L0MemoryOutMix.register_hook(self.variable_hook)
            #cv_pred_img
            #L0MemoryOutMix

            # evaluate lstm memory of l0, deconvolute l0 memory to image:
            # if gen_extradata and plotlevel >= 10:
            #     with torch.no_grad():
            #         if len(self.config.m_streams_spec) != 1:
            #             cv_l0_memory_content = self.cv_net.inspect_l0_memory(cv_next_state)
            #             cv_l0_memorystates.append(F.tanh(cv_l0_memory_content[:, vision_startIdx:]))
            #             cv_l1_memory_content = self.cv_net.inspect_l1_memory(cv_next_state)
            #             cv_l1_memorystates.append(F.tanh(cv_l1_memory_content[:, vision_startIdx:]))

            upprojected_cv_pred_control = self.transformer.window_to_image(cv_pred[:, 0:vision_startIdx], write_where,
                                                                           image_size=[self.config.v_dim,
                                                                                       self.config.v_dim])
            upprojected_cv_pred_img = self.transformer.window_to_image(cv_pred_img, write_where,
                                                                       image_size=[self.config.v_dim,
                                                                                   self.config.v_dim])

            upprojected_cv_pred = torch.cat([upprojected_cv_pred_control, upprojected_cv_pred_img], dim=1)
            # #calc loss to optimize local memory:
            # if self.L0Memory_trainsignal:
            #     #L0Memory_states[3]-(backgrounds*v_mask_states[1])
            #     v_targets_small = transformer.image_to_window(v_targets, attention_wheres)
            #     L0Mem_loss = ((L0Memory_states[3]-v_targets_small).pow(2)*mask_targets[:,:,None,None,None]*v_mask_states[1]).sum()/ input.numel()
            # mix between prediction or memory content for update of memory and prediction:
            # if self.pv_available:
            #     v_pred__mix_mask = upsample_pv_pred[:, 0:1]
            #     memory_mix_mask = upsample_pv_pred[:, 1:2]
            # else:
            v_pred__mix_mask = upprojected_cv_pred[:, 0:1]
            memory_mix_mask = upprojected_cv_pred[:, 1:2]

            upprojected_cv_image = upprojected_cv_pred[:, vision_startIdx:]
            fused_pv_cv = upprojected_cv_image  # no mixing, only one prediction!

            if self.hasLowLevelMemory:
                v_pred = (1. - v_pred__mix_mask) * canvas + v_pred__mix_mask * fused_pv_cv
                # update canvas if step >= num_context_frames:
                if step >= self.num_context_frames:
                    prev_canvas_filtered = (1. - v_pred__mix_mask) * canvas
                    canvas_self_feedback = (1. - memory_mix_mask) * canvas
                    # add hook to limit gradient flow to 1/2*1/2... N times "<1"
                    # canvas_self_feedback.register_hook(self.variable_hook)
                    canvas = canvas_self_feedback + memory_mix_mask * fused_pv_cv
            else:
                v_pred = fused_pv_cv

            i_incomings = []
            for l in range(len(self.config.integration_spec)):
                # ToDo multiple layers of lstm in integration network
                # assert (len(integration_states[-1]) == len(self.config.integration_spec))
                # integration_current_state.append(integration_states[-1][l])
                       # the lowest layer of integration network is lstm and the higher layers if any are pvrnn
                if self.is_bottomup:
                    # if self.is_pvrnn:
                    if l == len(self.config.integration_spec) - 1:   # the top pvrnn layer
                        i_incomings.append(
                            (cv_next_state[-1][0].view(n, -1), m_cv_next_state[-1][0]), )
                    else:
                        i_incomings = [[]]
                    # else:
                    #     if l == 0:  # the top  layer if using lstm
                    #         i_incomings.append(
                    #             (cv_next_state[-1][0].view(n, -1), m_cv_next_state[-1][0]),)
                else:
                    i_incomings = [[]]

            integration_next_state, mu_p, sigma_p, integration_gates, z = self.integration_net(i_incomings,
                                                                                               integration_current_state,
                                                                                               pv_mu,
                                                                                               pv_logvar, pv_mup_i, pv_logvar_i,
                                                                                               step=step)
            lang_pb_ = F.tanh(self.fc_integ_lang(integration_next_state[0][0]))
            # if self.is_lang:
            #     # if self.is_pb:
            #         # if self.is_integpb == False:
            #         #     lang_pb_ = F.tanh(self.fc_integ_lang(integration_next_state[0][0]))
            #             # lang_pb_ = nn.ReLU(self.fc_integ_lang(integration_next_state[0][0]))
            #         # else:
            #     lang_pb_ = torch.zeros(self.pb_dims).detach()
            lang_pb_preds.append(lang_pb_)
                # else:
                #     lang_init_ = F.tanh(self.fc_integ_lang(integration_next_state[0][0]))
                #     lang_init_preds.append(lang_init_)

            # if self.is_pvrnn:
            integration_state_gates.append(integration_gates)
            # integration_top_states.append(integration_next_state[-1][0])
            # mu_p_list.append(mu_p.detach().cpu().numpy())
            # sigma_p_list.append(sigma_p.detach().cpu().numpy())
            # mu_q_list.append(pv_mu.detach().cpu().numpy())
            # sigma_q_list.append(pv_logvar.detach().cpu().numpy())
            z_list.append(z)
            integration_states.append(integration_next_state)

            #compute pvrnn kld
            # if self.is_pvrnn:
            if self.training:
                for l in range(len(self.config.integration_spec)):
                    if step == 0:
                        pv_kl += self.w1[l] * kl_criterion(pv_mup_i[l], pv_logvar_i[l], pv_mu[l][:, step], pv_logvar[l][:, step])
                    else:
                        pv_kl += self.w[l] * (kl_criterion(mu_p[l], torch.log(sigma_p[l]), pv_mu[l][:, step], pv_logvar[l][:, step]))
            else:
                # just to test different metaprior values during planning
                for l in range(len(self.config.integration_spec)):
                    if step == 0:
                        pv_kl += self.w1[l] * kl_criterion(pv_mup_i[l], pv_logvar_i[l], pv_mu[l][:, step], pv_logvar[l][:, step])
                    else:
                        pv_kl += 10*self.w[l] * (kl_criterion(mu_p[l], torch.log(sigma_p[l]), pv_mu[l][:, step], pv_logvar[l][:, step]))
            v_states.append([cv_next_state])
            m_cv_states.append(m_cv_next_state)

            #add v9:
            # self.v_mask_memup_states_wnd.append(self.transformer.image_to_window(memory_mix_mask, write_where))
            self.v_mask_memup_states_wnd.append(memory_mix_mask)

            if step > self.num_context_frames - 2:

                cv_predictions.append(upprojected_cv_image)
                raw_cv_pred.append(cv_pred[:, vision_startIdx:])

                v_predictions.append(v_pred)
                if m_pred is not None: m_predictions.append(m_pred)


                memory_states.append(canvas)
                memory_feedback.append(prev_canvas_filtered)
                v_mask_memup_states.append(memory_mix_mask)  # cv_pred[:,1:2]
                v_mask_memout_states.append(v_pred__mix_mask)

                if self.useL0Memory:
                    if self.useL0Memory_v9 == True:
                        L0Memory_downsampled = F.interpolate(L0Memory,size=self.L0Memory_transformer_cv_dim)
                        L0Memory_states.append(L0Memory_downsampled)
                        if self.L0Memory_transformer is not None:
                            L0Memory_states_transformed.append(L0Memory_selection_downsampled)

                        L0Memory_states_feedback.append(L0MemoryMask)
                        L0Memory_states_outmix.append(L0MemoryOutMix)
                    else:
                        L0Memory_states.append(L0Memory)
                        if self.L0Memory_transformer is not None:
                            L0Memory_states_transformed.append(L0Memory_selection)
                        L0Memory_states_feedback.append(L0MemoryMask)
                        L0Memory_states_outmix.append(L0MemoryOutMix)
                # else:
                #     memory_states.append(canvas)
                #     memory_feedback.append(prev_canvas_filtered)

        # outputs are formed in batch first

        cv_predictions = torch.stack(cv_predictions, dim=1)

        v_predictions = torch.stack(v_predictions, dim=1)

        # if gen_evaldata or (self.useL0Memory_v9 == True):      # this part can be ignored
        #
        v_mask_mix_states = []
        if len(v_mask_memout_states) > 0:
            v_mask_memout_states = torch.stack(v_mask_memout_states, dim=1)
        if len(v_mask_memup_states) > 0:
            v_mask_memup_states = torch.stack(v_mask_memup_states, dim=1)
            self.v_mask_memup_states =v_mask_memup_states
        if len(self.v_mask_memup_states_wnd) > 0:
            self.v_mask_memup_states_wnd = torch.stack(self.v_mask_memup_states_wnd, dim=1)
        if len(memory_states) > 0:
            memory_states = torch.stack(memory_states, dim=1)
        if len(memory_feedback) > 0:
            memory_feedback = torch.stack(memory_feedback, dim=1)

        if self.useL0Memory:
            if len(L0Memory_states) > 0:
                L0Memory_states = torch.stack(L0Memory_states, dim=1)
            if len(L0Memory_states_feedback) > 0:
                L0Memory_states_feedback = torch.stack(L0Memory_states_feedback, dim=1)
            if len(L0Memory_states_outmix) > 0:
                L0Memory_states_outmix = torch.stack(L0Memory_states_outmix, dim=1)
            if len(L0Memory_states_transformed) > 0:
                L0Memory_states_transformed = torch.stack(L0Memory_states_transformed, dim=1)
            if len(L0Memory_updates) > 0:
                L0Memory_updates = torch.stack(L0Memory_updates, dim=1)
            if len(L0_memselection) > 0:
                L0_memselection = torch.stack(L0_memselection, dim=1)
            if len(L0_memmaskupscale) > 0:
                L0_memmaskupscale = torch.stack(L0_memmaskupscale, dim=1)
            if len(canvas_) > 0:
                canvas_ = torch.stack(canvas_, dim=1)
            if len(l0mem_updatemask) > 0:
                l0mem_updatemask = torch.stack(l0mem_updatemask, dim=1)
            if len(l0memupdatemask_pred) > 0:
                l0memupdatemask_pred = torch.stack(l0memupdatemask_pred, dim=1)

        v_mask_states = [v_mask_mix_states, v_mask_memout_states, v_mask_memup_states]

        if m_pred is not None:
            m_predictions = torch.stack(m_predictions, dim=1)
        else:
            m_predictions = None
        # if gen_extradata and plotlevel >= 10:
        #     if len(self.config.m_streams_spec) != 1:
        #         cv_l0_memorystates = torch.stack(cv_l0_memorystates, dim=1)
        #         cv_l1_memorystates = torch.stack(cv_l1_memorystates, dim=1)

        language_gates = []
        language_states = []
        if self.is_lang:
        # update language network:
            # lang_c = integration_states[-1][-1][0]
            # lang_h = F.tanh(lang_c)
            language_states = [lang_states]
            lang_init_states = []
            for l_step in range(len(language[0])):
                language_current_state = language_states[-1]
                if self.training:
                    if l_step > 0:
                        l = l_predictions[-1]
                    else:
                        l = language[:, l_step, :]
                else:
                    if gen_lang and l_step > 0:
                        l = l_predictions[-1]
                    else:
                        l = language[:, l_step, :]
                language_next_state, gates = self.language_net(l, language_current_state, lang_pb)
                # language_gates.append(gates)
                language_states.append(language_next_state[-1])
                l_pred = F.softmax(self.fc_l_pred(language_next_state[0][0]), dim=1)
                l_predictions.append(l_pred)

            l_predictions = torch.stack(l_predictions, dim=1)
        else:
            lang_states = [0]   # error if this list is empty
        attention_wheres = torch.stack(attention_wheres[self.num_context_frames:], dim=1)

        if self.training:
            # only output what is essential for training
            model_output = (m_predictions, v_predictions, cv_predictions, attention_wheres,
                             pv_kl, l_predictions, lang_pb_preds)
        else:
            model_output = m_predictions, v_predictions, cv_predictions, attention_wheres, \
                memory_states, memory_feedback, v_mask_states, \
                [L0Memory_states, L0Memory_states_outmix, L0Memory_states_feedback,
                 L0Memory_states_transformed, L0Memory_updates], \
                pv_kl, mu_p_list, sigma_p_list, mu_q_list, sigma_q_list, z_list, \
                l_predictions, m_cv_states, v_states, integration_state_gates, m_cv_state_gates, \
               cv_state_gates, language_gates, lang_states[0], lang_init_preds, lang_pb_preds, \
                [L0_memselection, L0_memmaskupscale, l0mem_updatemask, canvas_, l0memupdatemask_pred]

        return model_output
