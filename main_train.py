import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from opt import load_arg
from utils import IO, import_class
from models import Model_vision, masked_loss, kl_criterion, kl_fixed_logvar_criterion

import math
import numpy as np
import time
import itertools
import os
import pickle
import cv2
import uuid
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import random
# import numpy as np
# torch.manual_seed(12)
# torch.cuda.manual_seed_all(12)
# np.random.seed(12)
# random.seed(12)
# torch.backends.cudnn.deterministic=True

torch.set_num_threads(4)

is_half_precision=False

print("Working with torch ", torch.__version__)

if is_half_precision:
    torch.set_default_dtype(torch.float16)

args = load_arg()
device_ids = [args.device_ids] if isinstance(args.device_ids, int) else list(args.device_ids)


plotlevel= args.plotlevel #10-all, 5-low, 0-none

do_center_loss= args.model_args.get('do_center_loss', False)
do_sparse_memupdate_loss= args.model_args.get('do_sparse_memupdate_loss', False)


# intialize environment
io = IO(
    args.work_dir,
    save_log=args.save_log,
    print_log=args.print_log
)
io.save_arg(args)

# load_data
Feeder = import_class(args.feeder)
data_loader = dict()

train_feeder = Feeder(**args.train_feeder_args)
num_train_samples = len(train_feeder)
data_loader['train'] = DataLoader(
    dataset=train_feeder,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,#args.num_workers * len(device_ids),
    drop_last=False
)
# test_feeder = Feeder(**args.test_feeder_args)
# num_test_samples = len(test_feeder)
# data_loader['test'] = DataLoader(
#     dataset=test_feeder,
#     batch_size=args.test_batch_size,
#     shuffle=False,
#     num_workers=0#args.num_worker * ngpu(self.arg.device)
# )
l_seq_len, lang_dim = train_feeder[0][3].shape[0], train_feeder[0][3].shape[1]
seq_len, _ = train_feeder[0][2].shape[0], train_feeder[0][2].shape[1]

evalversion = args.evalversion

test_batch_size = args.test_batch_size
sample_start = args.sample_start
sample_num = args.sample_num
if args.sample_num_select<=0:
    samples_to_eval = list(range(sample_start, sample_start+sample_num))
else:
    step_ = float(sample_num-1)/float(args.sample_num_select)
    l1 = np.round(step_ * np.arange(0.0, float(args.sample_num_select)))
    samples_to_eval = list(l1.astype(int)+sample_start)
work_dir=args.work_dir

# load model
model = Model_vision(seed=args.seed, num_context_frames=args.num_context_frames, **args.model_args)
pb_size = args.model_args['language_args']['layers'][0]['pb_size']

lang_pb = []
lang_train_labels = []
loader1 = data_loader['train']

########################
# integ_pb = []
# for i in range(len(train_feeder)):
    # if args.model_args['integration_args']["is_integpb"]:
    #     integ_pb.append(nn.Parameter(torch.zeros(pb_size).clone().detach().cuda(args.cuda)))
########################
for indices, visions, motors, language, masks, lang_masks in loader1:
    lang = language.float()
    lang_train_labels.append(lang)
# lang_test_labels = []
# loader_t = data_loader['test']
# lang_t = torch.zeros((num_test_samples, l_seq_len, lang_dim))
# for indices, visions, motors, language, masks, lang_masks in loader_t:
#     lang = language.float()
#     lang_test_labels.append(lang)
#     lang_t[indices, :, :] = lang

# pca of language data used to initialize the initial states for language
for i in range(len(train_feeder)):
    if args.model_args['language_args']["is_lang"]:
        lang_pb.append(nn.Parameter(F.tanh(torch.zeros(pb_size).clone().detach().cuda(args.cuda))))
            # lang_pb.append(nn.Parameter(F.tanh(pca_pb_train[i].clone().detach().cuda(args.cuda))))  #tanh does nothing here

posterior_params = []
for _ in range(num_train_samples):
    posterior_param = {'mu': nn.Parameter(model.create_init_states()), 'logvar': nn.Parameter(model.create_init_states())}
    posterior_params.append(posterior_param)
#prior mu and logvar for initial step

pvrnn_pos_mu, pvrnn_pos_logvar = [], []
for n in range(num_train_samples):
    posterior_A_mu, posterior_A_logvar = model.pvrnn_init_states(args.model_args['integration_args']['layers'], seq_len=seq_len)
    pvrnn_pos_mu.append(posterior_A_mu)
    pvrnn_pos_logvar.append(posterior_A_logvar)

if args.model_args['integration_args']['is_UG'] == False:
    # learnable prior for initial state if not using unit gaussian
    pvrnn_prior_mu_i, pvrnn_prior_logvar_i = model.pvrnn_init_states(args.model_args['integration_args']['layers'],
                                                                     seq_len=1)
else:
    z_size = sum(args.model_args['integration_args']['layers'][l]['z_size'] for l in
                 range(len(args.model_args['integration_args']['layers'])))
    pvrnn_prior_mu_i, pvrnn_prior_logvar_i = torch.zeros(1, z_size), torch.zeros(1, z_size)

if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    # intention_params = checkpoint['intention']

v_criterion = nn.MSELoss(reduction='none')#size_average=False, reduce=False)      # vison loss
if args.model_args['motor_args'].get('is_softmax',True):
    m_criterion = nn.KLDivLoss(reduction='none')#(size_average=False, reduce=False)    # motor loss
else:
    m_criterion = nn.MSELoss(reduction='none')  # (size_average=False, reduce=False)    # motor loss
# a_criterion = nn.MSELoss(reduction='none')#(size_average=False, reduce=False)      # attention regularization loss
cv_center_loss_criterion = nn.MSELoss(reduction='none')
if args.lang_loss == 'mse':
    l_criterion = nn.MSELoss(reduction='none')
elif args.lang_loss =='bce':
    l_criterion = nn.BCELoss(reduction='none')
elif args.lang_loss =='kld':
    l_criterion = nn.KLDivLoss(reduction='none')
elif args.lang_loss =='ce':  #doesn't work I have to change language output activation
    l_criterion = nn.CrossEntropyLoss(reduction='none')

b_criterion = nn.MSELoss(reduction='mean')

# dev='cpu'
if args.use_gpu:
    dev = 'cuda:'+str(args.cuda)
else:
    dev = 'cpu'

model = model.to(dev)

if is_half_precision:
    model.half()
# intention_params = intention_params.to(dev)
v_criterion = v_criterion.to(dev)
m_criterion = m_criterion.to(dev)
l_criterion = l_criterion.to(dev)
b_criterion = b_criterion.to(dev)
cv_center_loss_criterion = cv_center_loss_criterion.to(dev)

if args.use_gpu:
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

# load optim
optimizer = dict()
optimizer['model'] = optim.Adam(
    model.parameters(),
    lr=args.base_lr,
    weight_decay=args.weight_decay
)

optimizer['pvrnn_posterior']= {
    'mu':
        [
        optim.Adam(
            [pvrnn_pos_mu[n]],
            lr=args.base_lr * 10 * 3,
            )for n in range(num_train_samples)
        ],
    'logvar':
        [
        optim.Adam(
            [pvrnn_pos_logvar[n]],
            lr=args.base_lr * 10 * 3,
            )for n in range(num_train_samples)
        ]}
if args.model_args['integration_args']['is_UG'] == False:
    optimizer['pvrnn_prior_i']= {
        'mu':
            optim.Adam(
                [pvrnn_prior_mu_i],
                lr=args.base_lr * 10 * 3,
                ),
        'logvar':
            optim.Adam(
                [pvrnn_prior_logvar_i],
                lr=args.base_lr * 10 * 3,
                )
            }


optimizer['lang_pb'] = \
{'lang_pb': [optim.Adam([lang_pb[n]],
                        lr=args.base_lr * 10 * 3,
                        ) for n in range(num_train_samples)]}


# get the coefficients from args
# beta = args.beta
K = args.k
target_reg_dynmodel = model.L0MemoryL1Reg
model.L0MemoryL1Reg=0
print(model)
def train(meta_info):
    model.train()
    # loader = data_loader['train']
    loss_value = []
    mse_value = []
    kl_value = []
    iter_info = dict()
    attention_prior = torch.Tensor([0,0,0]).view(1, 3).to(dev)
    #test distance to center:
    v_dim = args.model_args['vision_args']['dim']
    m1, m2 = torch.meshgrid([torch.arange(0, v_dim, dtype=torch.get_default_dtype()), torch.arange(0, v_dim, dtype=torch.get_default_dtype())])
    m1 = (m1/(v_dim-1.0)) #*2-1 1-...
    m2 = (m2/(v_dim-1.0)) #*2-1 1-..

    v_mesh = torch.stack([m2, m1], dim=2).to(dev)
    cv_center_loss = torch.tensor(0)
    sparse_memupdate_loss = torch.tensor(0)

    ib_state, integ_states = [], []
    l_pred, l_targ = [], []
    l_pb = []
    m_pred, m_targ = [], []
    ids = []
    epoloss= 0.
    epo_vloss = 0.
    epo_mloss = 0.
    epo_lloss = 0.
    epo_bloss = 0.
    epo_kld = 0.
    epo_pvkl = 0.

    for indices, visions, motors, language, masks, lang_masks in loader1:
        # do_eval_this_iter = (meta_info['iter'] % args.plot_interval == 0)
        time_start=time.time()
        # l_str = io.gen_lang(language[0])
        # print("lang = {}".format(l_str))
        indices = indices.long().to(dev)
        visions = visions.float().to(dev)
        for i in indices:
            ids.append(i)

        lang = language.float().to(dev)
        lang_masks = lang_masks.float().to(dev)
        l_inputs = lang[:, :-1]
        l_targets = lang[:, args.num_context_frames:]
        l_masks = lang_masks[:, :args.num_context_frames]

        if len(motors)>0: motors = motors.float().to(dev)
        masks = masks.float().to(dev)
        v_inputs = visions[:,:-1]

        m_inputs = []
        if len(motors) > 0: m_inputs = motors[:,:-1]
        v_targets = visions[:,args.num_context_frames:]
        m_targets = []
        if len(motors) > 0: m_targets = motors[:,args.num_context_frames:]
        mask_targets = masks[:,args.num_context_frames:]
        # setting posterior A values for each sequence
        pvrnn_posterior_mu = torch.stack([pvrnn_pos_mu[i] for i in indices], dim=0)
        pvrnn_posterior_logvar = torch.stack([pvrnn_pos_logvar[i] for i in indices], dim=0)
        posterior = {'mu': torch.stack([posterior_params[i]['mu'] for i in indices], dim=0),
                    'logvar': torch.stack([posterior_params[i]['logvar'] for i in indices], dim=0)}
        cell_mu = posterior['mu'].to(dev)
        cell_logvar = posterior['logvar'].to(dev)
        # cell_mu_prior = prior_param['mu'][None,:].to(dev)
        # cell_logvar_prior = prior_param['logvar'][None,:].to(dev)
        pv_mu = pvrnn_posterior_mu.to(dev)
        pv_logvar = pvrnn_posterior_logvar.to(dev)
        pv_prior_mu_i = pvrnn_prior_mu_i.to(dev)
        pv_prior_logvar_i = pvrnn_prior_logvar_i.to(dev)
        ediststart = 3750
        edistend = 5000
        edistdiff = (edistend - ediststart)
        if meta_info['epoch'] > ediststart:
            edistfac = min(float(meta_info['epoch'] - ediststart) / float(edistdiff), 1.0)
            # edistfac = 1
            model.L0MemoryL1Reg = target_reg_dynmodel * edistfac


        # if args.model_args['language_args']["is_lang"]:
        lang_pbs = torch.stack([lang_pb[i] for i in indices], dim=0)
        lang_pbs = lang_pbs.to(dev)

        model_out = model(v_inputs, m_inputs, l_inputs, cell_mu, cell_logvar, pv_mu, pv_logvar, pv_prior_mu_i,
                          pv_prior_logvar_i, lang_pb=lang_pbs)
        m_predictions, rv_predictions, cv_predictions, attention_wheres, \
            pv_kl, l_predictions, lang_pb_pred_list = model_out
        # else:
        #     model_out = model(v_inputs, m_inputs, l_inputs, cell_mu, cell_logvar, pv_mu, pv_logvar,
        #                       pv_prior_mu_i, pv_prior_logvar_i)
        #     m_predictions, rv_predictions, cv_predictions, attention_wheres, L0MemoryReg, init_states, v_mask_states, \
        #         pv_kl= model_out

        if args.model_args['language_args']["is_lang"]:
            l_pred.append(l_predictions.detach().cpu().numpy())
            l_targ.append(l_targets.detach().cpu().numpy())

        m_pred.append(m_predictions.detach().cpu().numpy())
        m_targ.append(m_targets.detach().cpu().numpy())

        v_loss = masked_loss(v_criterion, rv_predictions, v_targets, mask_targets)

        if len(motors) > 0: m_loss = masked_loss(m_criterion, m_predictions, m_targets, mask_targets)
        # kl_loss = kl_criterion(cell_mu, cell_logvar, cell_mu_prior, cell_logvar_prior)

        loss = 10*v_loss + 1*m_loss

        lossfactor = (float(meta_info['epoch']) / 500.0) - 1.0
        if lossfactor <= 0:
            bk = 1
        else:
            bk = K
        if args.model_args['language_args']["is_lang"]:
            lang_pbs = lang_pbs.repeat(len(v_inputs[0]), 1, 1)
            lang_pb_preds = torch.stack(lang_pb_pred_list)
            b_loss = b_criterion(lang_pb_preds, lang_pbs)

            # ib_state.append(lang_init_state.detach().cpu().numpy())
            for i in range(len(lang_pbs)):
                l_pb.append(lang_pbs[i].detach().cpu().numpy())
            l_loss = masked_loss(l_criterion, l_predictions, l_targets, l_masks)
            loss += K*l_loss + 1*b_loss
        else:
            l_loss = torch.zeros(1)
            b_loss = torch.zeros(1)

        # if args.model_args['integration_args']['is_pvrnn']:
        loss += pv_kl
        # else:
        #     pv_kl = torch.zeros(1)
        #     loss += beta * kl_loss

        # if len(L0MemoryReg)>0:        #regularization loss for VWM-2
        #     L0RegLoss =0
        #     for l in L0MemoryReg:
        #         L0RegLoss += l
        #     L0RegLoss = L0RegLoss / (len(L0MemoryReg)*len(mask_targets))
        #     iter_info['L0Memory_reg_loss'] = L0RegLoss.data.item()
        #     loss += L0RegLoss
        #loss for focus area
        if do_center_loss:
            #old: lossfactor = 1.0-(float(meta_info['iter'])/10000.0)
            lossfactor = 1.0 - (float(meta_info['epoch']) / 500.0)
            if lossfactor>0:
                cv_center = model.transformer.where_to_center(attention_wheres)
                dists = cv_center[:, :, None, None, :] - v_mesh
                distweight = torch.sum(torch.pow(dists, 2), dim=-1)
                prewdists = (rv_predictions - v_targets)
                wdists = prewdists.detach() * distweight[:, :, None, :, :]
                cv_center_loss = 100 * torch.mean(wdists ** 2)
                cv_center_loss=cv_center_loss*lossfactor
                loss += cv_center_loss

        #mem feedback norm loss
        # if do_sparse_memupdate_loss:
        #     sparse_memupdate_loss = 0.01 * torch.mean(torch.abs(v_mask_states[2]))
        #     loss += sparse_memupdate_loss

        optimizer['model'].zero_grad()
        # if args.model_args['integration_args']['is_pvrnn'] == False:
            # optimizer['prior']['mu'].zero_grad()
            # optimizer['prior']['logvar'].zero_grad()
        if args.model_args['integration_args']['is_UG'] == False:
            optimizer['pvrnn_prior_i']['mu'].zero_grad()
            optimizer['pvrnn_prior_i']['logvar'].zero_grad()
        for i in indices:
            optimizer['lang_pb']['lang_pb'][i].zero_grad()

            # if args.model_args['integration_args']['is_pvrnn']:
            optimizer['pvrnn_posterior']['mu'][i].zero_grad()
            optimizer['pvrnn_posterior']['logvar'][i].zero_grad()
            # else:
            #     optimizer['posterior']['mu'][i].zero_grad()
            #     optimizer['posterior']['logvar'][i].zero_grad()

        loss.backward()

        # gradiant clipping to avoid NaN
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        if args.model_args['language_args']['is_lang']:
            norm += nn.utils.clip_grad_norm_(lang_pb, args.clip_grad)

        # if args.model_args['integration_args']['is_pvrnn']:
        norm += nn.utils.clip_grad_norm_(pvrnn_pos_mu, args.clip_grad)
        norm += nn.utils.clip_grad_norm_(pvrnn_pos_logvar, args.clip_grad)
        if not args.model_args['integration_args']['is_UG']:
            norm += nn.utils.clip_grad_norm_(pvrnn_prior_mu_i, args.clip_grad)
            norm += nn.utils.clip_grad_norm_(pvrnn_prior_logvar_i, args.clip_grad)
        # else:
        #     norm += nn.utils.clip_grad_norm_(posterior_params[n]['mu'], args.clip_grad)
        #     norm += nn.utils.clip_grad_norm_(posterior_params[n]['logvar'], args.clip_grad)


        optimizer['model'].step()
        # if args.model_args['integration_args']['is_pvrnn'] == False:
        #     optimizer['prior']['mu'].step()
        #     optimizer['prior']['logvar'].step()
        if args.model_args['integration_args']['is_UG'] == False:
            optimizer['pvrnn_prior_i']['mu'].step()
            optimizer['pvrnn_prior_i']['logvar'].step()
        for i in indices:
            optimizer['lang_pb']['lang_pb'][i].step()
            # if args.model_args['integration_args']['is_pvrnn']:
            optimizer['pvrnn_posterior']['mu'][i].step()
            optimizer['pvrnn_posterior']['logvar'][i].step()
            # else:
            #     optimizer['posterior']['mu'][i].step()
            #     optimizer['posterior']['logvar'][i].step()

        # statistics
        epoloss += loss.data.item()
        epo_vloss += v_loss.data.item()
        epo_mloss += m_loss.data.item()
        epo_lloss += l_loss.data.item()
        epo_bloss += b_loss.data.item()
        # epo_kld += kl_loss.data.item()
        epo_pvkl += pv_kl.data.item()        # for plotting the kld without multiplying with w
        iter_info['loss'] = loss.data.item()
        iter_info['v_loss'] = v_loss.data.item()
        if args.model_args['language_args']['is_lang']:
            iter_info['l_loss'] = l_loss.data.item()
            iter_info['b_loss'] = b_loss.data.item()
        if len(motors) > 0: iter_info['m_loss'] = m_loss.data.item()
        # if args.model_args['integration_args']['is_pvrnn']:
        iter_info['pvkl_loss'] = pv_kl.data.item()
        # iter_info['kl_loss'] = kl_loss.data.item()
        iter_info['norm'] = norm#.data.item()
        iter_info['cv_center_loss']=cv_center_loss.data.item()
        iter_info['sparse_memupdate_loss']=sparse_memupdate_loss.data.item()
        iter_info['duration'] = time.time()-time_start
        # show_iter_info
        if meta_info['iter'] % args.log_interval == 0:
            info ='\tIter {} Done.'.format(meta_info['iter'])
            for k, v in iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            io.print_log(info)

        if args.model_args['language_args']["is_lang"]:
            mse_value.append(iter_info['v_loss'] + iter_info['m_loss']+ iter_info['l_loss'] + iter_info['b_loss'])
        else:
            mse_value.append(iter_info['v_loss'] + iter_info['m_loss'])
        # kl_value.append(iter_info['kl_loss'])
        loss_value.append(iter_info['loss'])

        meta_info['iter'] += 1

    return epo_vloss, epo_mloss, epo_lloss, epo_bloss, epo_kld, epo_pvkl

def start():
    io.print_log('Parameters:\n{}\n'.format(str(vars(args))))
    print("phase"+str(args.phase))
    # training phase
    #ToDo: generate loss curve at the end of training. Also include posterior reconstruction of trained data (for training accuracy)
    if args.phase == 'train':
        meta_info = dict(epoch=0, iter=0)
        if args.start_epoch>0:
            print("starting epoch = " + str(args.start_epoch))
            # load checkpoint:
            filename = 'epoch{}_checkpoint.pt'.format(args.start_epoch)
            # checkpoint = io.load_checkpoint(filename)
            # io.load_weights_from_checkpoint(model, checkpoint)
            # opti_model = io.load_optimizer(checkpoint, param='model') #io.load_optimizer_model(checkpoint)
            # if args.model_args['language_args']["is_lang"]:
                # lang_pb_load = io.load_from_checkpoint(checkpoint, param='lang_pb')

            # if args.model_args['integration_args']['is_pvrnn']:
            # pvrnn_pos_mu_load = io.load_from_checkpoint(checkpoint, param='pvrnn_pos_mu')#io.load_pvrnn_posterior_mu_from_checkpoint(checkpoint)
            # pvrnn_pos_logvar_load = io.load_from_checkpoint(checkpoint, param='pvrnn_pos_logvar')#io.load_pvrnn_posterior_logvar_from_checkpoint(checkpoint)
            # opti_pvrnn_posterior = io.load_optimizer(checkpoint, param='pvrnn_posterior') #io.load_optimizer_pvrnn_posterior(checkpoint)
            # if args.model_args['integration_args']['is_UG'] == False:
            #     pvrnn_prior_mu_i_load = io.load_prior_from_checkpoint(checkpoint, param='pvrnn_prior_mu_i')
            #     pvrnn_prior_logvar_i_load = io.load_prior_from_checkpoint(checkpoint, param='pvrnn_prior_logvar_i')
            # else:
            #     posterior_params_load = io.load_from_checkpoint(checkpoint, param='posterior') #io.load_posteriors_from_checkpoint(checkpoint)
            #     prior_param_load = io.load_from_checkpoint(checkpoint, param='prior') #io.load_prior_from_checkpoint(checkpoint)
            #     opti_prior = io.load_optimizer(checkpoint, param='prior') #io.load_optimizer_prior(checkpoint)
            #     opti_posterior = io.load_optimizer(checkpoint, param='posterior') #io.load_optimizer_posterior(checkpoint)

            # optimizer['model'].load_state_dict(opti_model)
            # if len(posterior_params_load) < num_train_samples:
            #     for i in range(num_train_samples-len(posterior_params_load)):
            #         intention_param = {'mu': nn.Parameter(model.create_init_states()),
            #                            'logvar': nn.Parameter(model.create_init_states())}
            #
            #         intention_pvrnn_mu, intention_pvrnn_logvar = model.pvrnn_init_states(
            #             args.model_args['integration_args']['layers'], seq_len=seq_len)
            #
            #         pvrnn_pos_mu_load.append(intention_pvrnn_mu)
            #         pvrnn_pos_logvar_load.append(intention_pvrnn_logvar)
            #
            # for s_nr in range(num_train_samples-len(posterior_params_load)):
            #     # set prior ? from learning as initial guess:
            #     posterior_params[s_nr]['mu'].data.copy_(posterior_params_load[s_nr]['mu'].data)
            #     posterior_params[s_nr]['logvar'].data.copy_(posterior_params_load[s_nr]['logvar'].data)
            #     pvrnn_pos_mu[s_nr].data.copy_(pvrnn_pos_mu_load[s_nr].data)
            #     pvrnn_pos_logvar[s_nr].data.copy_(pvrnn_pos_logvar_load[s_nr].data)
            #     # if args.model_args['integration_args']['is_pvrnn']:
            #     optimizer['pvrnn_posterior']['mu'][s_nr].load_state_dict(opti_pvrnn_posterior[0][s_nr])
            #     optimizer['pvrnn_posterior']['logvar'][s_nr].load_state_dict(opti_pvrnn_posterior[1][s_nr])
            #         # prior_param['mu'].data.copy_(prior_param_load['mu'].data)
                    # prior_param['logvar'].data.copy_(prior_param_load['logvar'].data)
                # else:
                #     optimizer['posterior']['mu'][s_nr].load_state_dict(opti_posterior[0][s_nr])
                #     optimizer['posterior']['logvar'][s_nr].load_state_dict(opti_posterior[1][s_nr])
                #     optimizer['prior']['mu'].load_state_dict(opti_prior[0])
                #     optimizer['prior']['logvar'].load_state_dict(opti_prior[1])

        FE= []
        vision_error, motor_error, lang_error, bind_error, kld, pv_kl = [], [], [], [], [], []
        for epoch in range(args.start_epoch, args.num_epochs):
            meta_info['epoch'] = epoch
            epoch_start = time.time()
            # training
            # if args.model_args['language_args']["is_lang"]:
            v_loss, m_loss, l_loss, b_loss, kl, pvkl = train(meta_info)
            # else:
            #     v_loss, m_loss, l_loss, b_loss, kl, pvkl = train(meta_info)
            loss_value = v_loss + m_loss + l_loss + b_loss + pvkl
            FE.append(loss_value)
            vision_error.append(v_loss)
            motor_error.append(m_loss)
            lang_error.append(l_loss)
            bind_error.append(b_loss)
            kld.append(kl)
            pv_kl.append(pvkl)
            io.print_log('Training epoch: {}| loss: {:.4f} | v_loss: {:.4f}| m_loss: {:.4f}| l_loss: {:.4f}| b_loss:{:.4f}| pv_kl: {:.4f}'.format(
                epoch, loss_value, v_loss, m_loss, l_loss, b_loss, pvkl))
            io.print_log('Done. Duration: {}'.format(time.time() - epoch_start))
            # save model
            if ((epoch + 1) % args.save_interval == 0) or (epoch + 1 == args.num_epochs):
                # import pdb
                # pdb.set_trace()
                filename = 'epoch{}_checkpoint.pt'.format(epoch + 1)
                save_dict = {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'loss_value': loss_value,
                    'optimizer': {'model': optimizer['model'].state_dict()},
                }
                # if args.model_args['integration_args']['is_pvrnn']:
                save_dict['pvrnn_pos_mu'] = pvrnn_pos_mu
                save_dict['pvrnn_pos_logvar'] = pvrnn_pos_logvar
                save_dict['optimizer']['pvrnn_posterior'] = [[o.state_dict() for o in optimizer['pvrnn_posterior']['mu']],
                                                [o.state_dict() for o in
                                                 optimizer['pvrnn_posterior']['logvar']]]
                if not args.model_args['integration_args']['is_UG']:
                    save_dict['pvrnn_prior_mu_i'] = pvrnn_prior_mu_i
                    save_dict['pvrnn_prior_logvar_i'] = pvrnn_prior_logvar_i
                    save_dict['optimizer']['pvrnn_prior_i'] = optimizer['pvrnn_prior_i']
                # else:
                    # save_dict['prior'] = prior_param
                    # save_dict['posterior'] = posterior_params
                    # save_dict['optimizer']['prior'] = [optimizer['prior']['mu'].state_dict(),
                    #                                   optimizer['prior']['logvar'].state_dict()]
                    # save_dict['optimizer']['posterior'] = [[o.state_dict() for o in optimizer['posterior']['mu']],
                    #                                        [o.state_dict() for o in optimizer['posterior']['logvar']]]

                # if args.model_args['language_args']["is_lang"]:
                save_dict['lang_train_labels'] = lang_train_labels
                    # if args.model_args['language_args']["is_pb"]:
                save_dict['lang_pb'] = lang_pb
                save_dict['optimizer']['lang_pb'] = [o.state_dict() for o in optimizer['lang_pb']['lang_pb']]
                    # else:
                    #     save_dict['lang_init_state'] = lang_init
                    #     save_dict['optimizer']['lang_init_state'] = [o.state_dict() for o in
                    #                                          optimizer['lang_init_state']['lang_init_state']]
                loss_dict = {'v_loss': vision_error, 'm_loss': motor_error, "l_loss": lang_error,
                             "b_loss": bind_error, "pv_kl": pv_kl, 'loss': FE}
                np.savez(work_dir + "/loss_{}".format(epoch), **loss_dict)

                io.save_checkpoint(save_dict, filename)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    start()
