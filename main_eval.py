import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from opt import load_arg
from utils import IO, import_class
from models import Model_vision, masked_loss, kl_criterion, kl_fixed_logvar_criterion


import time

import itertools

import cv2

import pickle
import os

import numpy as np

import uuid

from sklearn.decomposition import PCA


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
    num_workers=0,
    drop_last=False
)
test_feeder = Feeder(**args.test_feeder_args)
num_test_samples = len(test_feeder)
data_loader['test'] = DataLoader(
    dataset=test_feeder,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=0#
)

l_seq_len, lang_dim = test_feeder[0][3].shape[0], test_feeder[0][3].shape[1]
seq_len, _ = test_feeder[0][2].shape[0], test_feeder[0][2].shape[1]
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
lang_init_dim = lang_dim
######################## using pca of language data to initialize language initial states for training and testing
loader1 = data_loader['train']
lang_i = torch.zeros((num_train_samples, l_seq_len, lang_dim))
for indices, visions, motors, language, masks, lang_masks in loader1:
    lang = language.float()
    lang_i[indices, :, :] = lang

loader_t = data_loader['test']
lang_t = torch.zeros((num_test_samples, l_seq_len, lang_dim))
for indices, visions, motors, language, masks, lang_masks in loader_t:
    lang = language.float()
    lang_t[indices, :, :] = lang

nc_i = args.model_args['language_args']['layers'][0]['hid_size'] #dimenstion of language d neurons
pca_i = PCA(n_components=nc_i)
nc_pb = args.model_args['language_args']['layers'][0]['pb_size']
pca_pb = PCA(n_components=nc_pb)

def pca_init(train_lang, test_lang, pca):
    lang = torch.cat((train_lang, test_lang))
    pca_lang = pca.fit_transform(lang.numpy().reshape(len(lang), -1))
    train_lang_pca, test_lang_pca = pca_lang[:len(train_lang), :], pca_lang[len(train_lang): , :]
    return torch.from_numpy(train_lang_pca), torch.from_numpy(test_lang_pca)
pca_lang, pca_lang_test = pca_init(lang_i, lang_t, pca_i)
pca_pb_train, pca_pb_test = pca_init(lang_i, lang_t, pca_pb)

lang_init_test = []
lang_pb_test = []
for i in range(len(pca_lang_test)):
    if args.model_args['language_args']["is_pb"]:
        lang_init_test.append(torch.zeros_like(pca_lang_test[i].clone().detach().cuda(args.cuda)))
        lang_pb_test.append(torch.zeros(pb_size).clone().detach().cuda(args.cuda))
        # lang_pb_test.append(F.tanh(pca_pb_test[i].clone().detach().cuda(args.cuda)))  # tanh does nothing here

    else:
        lang_init_test.append(pca_lang_test[i].clone().detach().cuda(args.cuda))
        lang_pb_test.append(torch.zeros(pb_size).clone().detach().cuda(args.cuda))

######################################################################################################################

#create prior for cell states (mu and log var)
prior_param = {'mu': nn.Parameter(model.create_init_states()), 'logvar': nn.Parameter(model.create_init_states())}

posterior_params = []
for _ in range(num_train_samples):
    posterior_param = {'mu': nn.Parameter(model.create_init_states()), 'logvar': nn.Parameter(model.create_init_states())}
    posterior_params.append(posterior_param)

#prior mu and logvar for initial step
if args.model_args['integration_args']['is_UG'] == False:
    pvrnn_prior_mu_i, pvrnn_prior_logvar_i = model.pvrnn_init_states(args.model_args['integration_args']['layers'], seq_len=1)
else:
    pvrnn_prior_mu_i, pvrnn_prior_logvar_i = torch.zeros(args.model_args['integration_args']['layers'][-1]['z_size']),\
        torch.zeros(args.model_args['integration_args']['layers'][-1]['z_size'])

pvrnn_pos_mu, pvrnn_pos_logvar = [], []
for n in range(num_train_samples):
    posterior_A_mu, posterior_A_logvar = model.pvrnn_init_states(args.model_args['integration_args']['layers'], seq_len=seq_len)
    pvrnn_pos_mu.append(posterior_A_mu)
    pvrnn_pos_logvar.append(posterior_A_logvar)

if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])

if args.lang_loss == 'mse':
    l_criterion = nn.MSELoss(reduction='none')
elif args.lang_loss =='bce':
    l_criterion = nn.BCELoss(reduction='none')
elif args.lang_loss =='kld':
    l_criterion = nn.KLDivLoss(reduction='none')
elif args.lang_loss =='ce': #doesn't work
    l_criterion = nn.CrossEntropyLoss(reduction='none')
v_criterion = nn.MSELoss(reduction='none')     # vison loss
if args.model_args['motor_args'].get('is_softmax',True):
    m_criterion = nn.KLDivLoss(reduction='none')    # motor loss
else:
    m_criterion = nn.MSELoss(reduction='none')   # motor loss
a_criterion = nn.MSELoss(reduction='none')    # attention regularization loss # not used

cv_center_loss_criterion = nn.MSELoss(reduction='none')

b_criterion = nn.MSELoss(reduction='mean')             # binding loss

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

# get the coefficients from args
beta = args.beta

K = args.k


def test(pv_posterior=None, pv_prior=None, lang_pb=None, prefix='', pred_lang=False, vision_goal=False):
    # print(model)
    model.eval()
    loader = data_loader['test']
    loss_value = []
    mse_value, kl_value = [], []
    vision_loss, motor_loss, lang_loss = [], [], []  #for evaluating prediction accuracy
    iter_info = dict()
    sample_size = 1
    folder='eval'

    samples_to_eval
    target_intention_params = []
    target_pvrnn_intention_mu, target_pvrnn_intention_logvar = [], []
    target_pvrnn_intention_mu_i, target_pvrnn_intention_logvar_i = [], []
    target_intention_lang_inits = []
    target_intention_pbs = []
    target_intention_integpbs = []
    lang_labels = []
    lang_latent = []
    for s_nr in samples_to_eval:
        indices_, visions_, motors_, language_, masks_, l_masks_ = test_feeder.__getitem__(
            s_nr)  # assume only one sample at a time!!
        seq_len = len(visions_)

        intention_pvrnn_mu, intention_pvrnn_logvar = model.pvrnn_init_states(
            args.model_args['integration_args']['layers'], seq_len=seq_len)
        intention_param = {'mu': nn.Parameter(model.create_init_states()),
                           'logvar': nn.Parameter(model.create_init_states())}

        if args.model_args['integration_args']['is_UG'] == False:
            intention_pv_prior_mu, intention_pv_prior_logvar = pv_prior['mu'], pv_prior['logvar']
        else:
            intention_pv_prior_mu, intention_pv_prior_logvar = \
                nn.Parameter(pvrnn_prior_mu_i.data.clone()), nn.Parameter(pvrnn_prior_logvar_i.data.clone())
        if args.model_args['language_args']["is_lang"]:
            if lang_pb is not None:
                intention_pb = nn.Parameter(torch.zeros(pb_size))
                # intention_pb = nn.Parameter(lang_pb_test[s_nr].data.clone())
                # if args.model_args['integration_args']["is_integpb"]:
                #     intention_integpb = nn.Parameter(torch.zeros(pb_size))
            # if lang_init is not None:
            #     # intention_lang_inits = nn.Parameter(torch.zeros(lang_init_dim))
            #     intention_lang_inits = nn.Parameter(lang_init_test[s_nr].data.clone())

        target_pvrnn_intention_mu.append(intention_pvrnn_mu)
        target_pvrnn_intention_logvar.append(intention_pvrnn_logvar)
        target_pvrnn_intention_mu_i.append(intention_pv_prior_mu)
        target_pvrnn_intention_logvar_i.append(intention_pv_prior_logvar)
        target_intention_params.append(intention_param)

        if args.model_args['language_args']["is_lang"]: # for plotting
            if lang_pb is not None:
                target_intention_pbs.append(intention_pb)
                lang_latent.append(intention_pb.detach().cpu())

    stepwidth=test_batch_size
    # su, v_su, m_su, l_su = 0, 0, 0, 0  # successes
    # fa, v_f, m_f, l_f = 0, 0, 0, 0      # failures
    for u in range(0,len(samples_to_eval), stepwidth):
        samples = samples_to_eval[u:u+stepwidth]
        indices=[]
        indices_local = list(range(stepwidth))
        visions=[]
        motors=[]
        masks=[]
        language = []
        lang_masks = []
        foldernames=[]
        optimization_results=[]
        for i in range(stepwidth):
            ndict = {'min_loss': -1, 'losses': [], 'configs': [], 'min_config': []}
            for t_i in range(sample_size):
                ndict['losses'].append([])
                ndict['configs'].append([])
            optimization_results.append(ndict)

        for s_nr in samples:
            indices_, visions_, motors_, language_, masks_, lang_masks_ = test_feeder.__getitem__(s_nr)  # assume only one sample at a time!!
            indices.append(torch.tensor(indices_ - args.sample_start))
            visions.append(torch.tensor(visions_))
            motors.append(torch.tensor(motors_))
            masks.append(torch.tensor(masks_))
            language.append(torch.tensor(language_))
            lang_masks.append(torch.tensor(lang_masks_))
            print("sample number="+str(s_nr))
            nfolder = prefix+'sample'+str(s_nr)
            try:
                os.makedirs(work_dir + nfolder)
            except FileExistsError:
                None
            foldernames.append(nfolder)

        indices = torch.stack(indices)
        visions = torch.stack(visions)
        motors = torch.stack(motors)
        masks = torch.stack(masks)
        language = torch.stack(language)
        lang_masks = torch.stack(lang_masks)
        lang_mask = lang_masks.to(dev)
        lang_mask = lang_mask.repeat(sample_size, 1)
        lang = language.float().to(dev)
        lang_labels.append(lang.detach().cpu())
        lang = lang.repeat(sample_size, 1, 1)
        l_inputs = lang[:, :-1]
        l_targets = lang[:, args.num_context_frames:]

        indices = indices.long().to(dev)
        visions = visions.float().to(dev)
        visions = visions.repeat(sample_size, 1, 1, 1, 1)
        if len(motors)>0:
            motors = motors.float().to(dev)
            motors = motors.repeat(sample_size, 1, 1)
        masks = masks.float().to(dev)
        masks = masks.repeat(sample_size, 1)
        v_inputs = visions[:,:-1]

        m_inputs = []
        if len(motors) > 0: m_inputs = motors[:,:-1]

        v_targets = visions[:,args.num_context_frames:]

        m_targets = []
        if len(motors) > 0: m_targets = motors[:,args.num_context_frames:]
        lang_mask_targets = lang_mask[:, args.num_context_frames:]
        mask_targets = masks[:,args.num_context_frames:]
        length = masks.sum(1).byte()
        length_idxs = list(length.cpu().numpy() - 2)

        regression_optimizer = {}         # optimizer for error regression / goal directed planning
        # if args.model_args['integration_args']["is_pvrnn"]:
        regression_optimizer['pv_mu'] = optim.Adam(
            [target_pvrnn_intention_mu[i] for i in indices],
            lr=args.base_lr * 10 * 30,
        )
        regression_optimizer['pv_logvar'] = optim.Adam(
            [target_pvrnn_intention_logvar[i] for i in indices],
            lr=args.base_lr * 10 * 30,
        )
        if args.model_args['integration_args']['is_UG'] == False:
            regression_optimizer['pvrnn_prior_i']= {
                'mu': optim.Adam(
                [intention_pv_prior_mu],
                lr=args.base_lr * 10 * 50,
                ),
            'logvar': optim.Adam(
                [intention_pv_prior_logvar],
                lr=args.base_lr * 10 * 50,
                )
            }

        regression_optimizer['lang_pb'] = optim.Adam([target_intention_pbs[i] for i in indices],
                                    lr=args.base_lr * 10 * 30,
                                    )

        indices_rep = indices.repeat(sample_size)
        indices_local_rep=torch.tensor(indices_local).repeat(sample_size)

        for step in range(args.num_regressions):
            # print(indices) # number of iterations of error regression, we want to get the interation index with the lowest prediction error in all modalities
            pvrnn_posterior_mu = torch.stack([target_pvrnn_intention_mu[i] for i in indices], dim=0)
            pvrnn_posterior_logvar = torch.stack([target_pvrnn_intention_logvar[i] for i in indices], dim=0)
            pv_mu = pvrnn_posterior_mu.to(dev)
            pv_logvar = pvrnn_posterior_logvar.to(dev)
            #repeat n times for n parallel optimizations!
            # repeat copies tensor!
            pv_mu = pv_mu.repeat(sample_size, 1, 1)
            pv_logvar = pv_logvar.repeat(sample_size, 1, 1)
            pv_prior_mu_ = intention_pv_prior_mu.to(dev)
            pv_prior_logvar_ = intention_pv_prior_logvar.to(dev)
            intention = {'mu': torch.stack([target_intention_params[i]['mu'] for i in indices], dim=0),
                         'logvar': torch.stack([target_intention_params[i]['logvar'] for i in indices], dim=0)}
            intention_mu = intention['mu'].to(dev)
            intention_logvar = intention['logvar'].to(dev)
            intention_mu = intention_mu.repeat(sample_size, 1)
            intention_logvar = intention_logvar.repeat(sample_size, 1)


            if args.model_args['language_args']["is_lang"]:
                # if args.model_args['integration_args']["is_integpb"]:
                #     integ_pbs = torch.stack([target_intention_integpbs[i] for i in indices], dim=0)
                #     integ_pbs = integ_pbs.repeat(sample_size, 1)
                #     integ_pbs = integ_pbs.to(dev)
                # else:
                #     integ_pbs = None

                if args.model_args['language_args']["is_pb"]:
                    lang_pbs = torch.stack([target_intention_pbs[i] for i in indices], dim=0)
                    lang_pbs = lang_pbs.repeat(sample_size, 1)
                    lang_pbs = lang_pbs.to(dev)
                    lang_intention=None
                # else:
                #     lang_intention = torch.stack([target_intention_lang_inits[i] for i in indices], dim=0)
                #     lang_intention = lang_intention.repeat(sample_size, 1)
                #     lang_intention = lang_intention.to(dev)
                #     lang_pbs = None

            do_eval_this_iter = (step % 10 == 1)
            t_start = time.time()

            if args.model_args['language_args']["is_lang"]:
                (m_predictions, rv_predictions, cv_predictions, attention_wheres, backgrounds, \
                memory_feedback, v_mask_states, L0Memory_states,\
                pv_kl, mu_p, sigma_p, mu_q, sigma_q, z,\
                l_predictions, m_cv_state, v_state, integration_state_gates, m_cv_state_gates,
                 cv_state_gates, language_state_gates,\
                lang_init_state, lang_init_pred_list, lang_pb_pred_list, l0states)\
                    = model(v_inputs, m_inputs, l_inputs, intention_mu, intention_logvar,
                                              pv_mu, pv_logvar, pv_prior_mu_, pv_prior_logvar_, lang_pbs, gen_lang=pred_lang)

                model.transformer.where_on_image(attention_wheres, rv_predictions)   # to get the attention box in prediction

                if args.model_args['language_args']['is_pb']:
                    lang_pbs = lang_pbs.repeat(len(v_inputs[0]), 1, 1)
                    lang_pb_preds = torch.stack(lang_pb_pred_list)

                    b_loss = b_criterion(lang_pb_preds, lang_pbs)

                l_loss = masked_loss(l_criterion, l_predictions, l_targets, lang_mask_targets)                # language loss
                if len(motors) > 0: m_loss = masked_loss(m_criterion, m_predictions, m_targets, mask_targets) # motor loss
                v_loss = masked_loss(v_criterion, rv_predictions, v_targets, mask_targets)                    # vision loss

                _loss = 0          # reconstruction loss of goal minimized for planning
                plot_losses = []    # loss used to get the results for plotting | this is not optimized
                if pred_lang: # generate language when vision and motor are provided
                    for i_loss in range(v_targets.shape[0]):   #v_targets.shape[0] = batchsize which is always 1 for now
                        closs = 1*v_loss  #F.mse_loss(rv_predictions[i_loss, :], v_targets[i_loss, :])  # , mask_targets)
                        closs += 1*m_loss
                        closs += 1*b_loss
                        _loss += closs
                        plot_losses.append(closs.data.item())
                        # plot_losses[i_loss] += v_loss.data.item() #F.mse_loss(rv_predictions[i_loss, :],
                                                          #v_targets[i_loss, :])
                        # plot_losses[i_loss] += m_loss.data.item() #masked_loss(m_criterion, m_predictions, m_targets, mask_targets)
                        plot_losses[i_loss] += l_loss.data.item()
                        # plot_losses[i_loss] += b_loss.data.item()

                elif vision_goal: # goal image used for planning (Jeff's model)
                    for i_loss in range(v_targets.shape[0]):
                        closs = F.mse_loss(rv_predictions[i_loss, length_idxs[i_loss]], v_targets[i_loss, length_idxs[i_loss]])
                        print("g_loss = {}".format(closs.data.item()))
                        plot_losses.append(closs.data.item())
                        closs += F.mse_loss(rv_predictions[i_loss, :3], v_targets[i_loss, :3])  # loss for the first few steps of vision
                        _loss += closs
                        _loss += b_loss
                        plot_losses[i_loss] += v_loss.data.item()
                        plot_losses[i_loss] += m_loss.data.item()
                        plot_losses[i_loss] += l_loss.data.item()
                        plot_losses[i_loss] += b_loss.data.item()

                else:
                    # language goal
                    for i_loss in range(v_targets.shape[0]):
                        closs = K*l_loss #masked_loss(l_criterion, l_predictions, l_targets, lang_mask_targets) #l_criterion(l_predictions[i_loss], l_targets[i_loss])
                        closs += 10*b_loss
                        closs += 1*F.mse_loss(rv_predictions[i_loss, :3], v_targets[i_loss, :3])  # , mask_targets)
                        plot_losses.append(l_loss.data.item())

                        plot_losses[i_loss] += v_loss.data.item() #F.mse_loss(rv_predictions[i_loss, :],
                                                         # v_targets[i_loss, :])
                        plot_losses[i_loss] += m_loss.data.item() #masked_loss(m_criterion, m_predictions, m_targets, mask_targets)
                        plot_losses[i_loss] += b_loss.data.item()
                        _loss += closs

                loss = _loss + pv_kl
                # else:
                #     loss = _loss + beta * kl_loss
                print("v_loss = {:.6f} | m_loss = {:.6f} | l_loss = {:.6f} |b_loss = {:.6f} | pv_kl = {:.6f}".format(
                    v_loss.data.item(), m_loss.data.item(), l_loss.data.item(), b_loss.data.item(), pv_kl.data.item()))
            else:
                # no language
                m_predictions, rv_predictions, cv_predictions, attention_wheres, backgrounds, \
                memory_feedback, v_mask_states, cv_ln_memorystates, L0Memory_states, L0MemoryReg,\
                initstates, int_states, pv_kl, mu_p, sigma_p, mu_q, sigma_q, z,\
                l_predictions, m_cv_state, v_state, integration_state_gates, m_cv_state_gates,\
                 cv_state_gates, _,\
                _, _,  _, l0states\
                    = model(v_inputs, m_inputs, l_inputs, intention_mu, intention_logvar,
                                              pv_mu, pv_logvar, pv_prior_mu_, pv_prior_logvar_)

                model.transformer.where_on_image(attention_wheres, rv_predictions)

                if len(motors) > 0: m_loss = masked_loss(m_criterion, m_predictions, m_targets, mask_targets)
                v_loss = masked_loss(v_criterion, rv_predictions, v_targets, mask_targets)  # vision loss
                l_loss = torch.zeros(1)

                #Tg loss
                # Tg=1
                _loss = 0         # reconstruction loss of goal minimized for planning
                plot_losses = []   # loss used to get the results for plotting
                for i_loss in range(v_targets.shape[0]):
                    closs = F.mse_loss(rv_predictions[i_loss,length_idxs[i_loss]], v_targets[i_loss,length_idxs[i_loss]]) # vision goal loss
                    plot_losses.append(closs.data.item())
                    plot_losses[i_loss] += closs.data.item()
                    _loss += closs
                    closs += F.mse_loss(rv_predictions[i_loss,:4], v_targets[i_loss,:4]) # vision loss from first few steps
                    plot_losses[i_loss] += closs.data.item()
                    _loss += closs

                    loss = _loss + pv_kl

                print("v_loss:{:.6f} | m_loss:{:.6f} | l_loss:{:.6f} | pv_kl:{:.6f}".format(
                    v_loss.data.item(), m_loss.data.item(), l_loss.data.item(), pv_kl.data.item()))
            # save log of states:

            plot_idxs=[]
            for i_loss in range(v_targets.shape[0]):
                #i_loss = batch number for multiple batches? here it does nothing, need to make this work for multiple batches

                c_loss = plot_losses[i_loss]
                c_cfg = {'mu': intention_mu[i_loss].cpu().detach().numpy(),
                         'logvar': intention_logvar[i_loss].cpu().detach().numpy(),
                         'pv_mu': pv_mu[i_loss].cpu().detach().numpy(),
                         'pv_logvar': pv_logvar[i_loss].cpu().detach().numpy(),
                         'step': step,
                         'length': length_idxs[i_loss]}
                c_min_loss = optimization_results[indices_local_rep[i_loss]]['min_loss']
                # print("iterloss = {}".format(c_loss))
                if c_min_loss<0 or c_min_loss>c_loss:
                    #update best solution!
                    print("idx: {} min_loss={}".format(i_loss, c_min_loss))
                    plot_idxs.append(i_loss)
                    optimization_results[indices_local_rep[i_loss]]['min_loss'] = plot_losses[i_loss] # m_loss + vis_loss + l_loss # total reconstruction loss
                    optimization_results[indices_local_rep[i_loss]]['min_config']=c_cfg

                    motor_prediciton = (m_predictions[i_loss].detach().cpu())
                    vision_predictions = (rv_predictions[i_loss, :, :, :].detach().cpu() + 1) * 0.5
                    vision_targets = (v_targets[i_loss, :, :, :].detach().cpu() + 1) * 0.5
                    cv_prediction = (cv_predictions[i_loss, :, :, :].detach().cpu() + 1) * 0.5


                    l0mem = None
                    l0mem_outmix = None
                    l0mem_memmix = None
                    maskedl0mem = None
                    l0mem_updatesource_ = None
                    l0mem_selection = None
                    l0mem_maskupscale = None
                    l0mem_updatemask = None
                    l0memupdatemask_pred = None
                    wnds = None
                    canvas_ = None
                    if len(L0Memory_states[0]) > 0:
                        l0mem = L0Memory_states[0][i_loss].cpu().detach()
                        l0mem_outmix = L0Memory_states[1][i_loss].cpu().detach().numpy()
                        l0mem_memmix = L0Memory_states[2][i_loss].cpu().detach().numpy()
                        l0mem_selection = ((l0states[0][i_loss] + 1) * 0.5 ).cpu().detach()
                        l0mem_maskupscale = ((l0states[1][i_loss] + 1) * 0.5).cpu().detach()
                        l0mem_updatemask = ((l0states[2][i_loss] + 1) * 0.5).detach().cpu()
                        l0memupdatemask_pred = ((l0states[4][i_loss] + 1) * 0.5).detach().cpu()
                        canvas_ = ((l0states[3][i_loss] + 1) * 0.5).detach().cpu()
                        wnds = model.transformer.image_to_window(v_mask_states[1][i_loss], attention_wheres[i_loss])

                        maskedl0mem = ((((l0mem + 1) * 0.5) * wnds.cpu())).detach().cpu()
                        wnds = wnds.detach().cpu().numpy()
                        l0mem_updatesource_ = ((L0Memory_states[4][i_loss] + 1) * 0.5).detach().cpu()

                        l0mem = l0mem.numpy()
                        maskedl0mem = maskedl0mem.numpy()


                    bg = backgrounds[i_loss].clone().detach().cpu().detach()

                    v_mix_mask_refresh = v_mask_states[2][0].detach().data.cpu()
                    v_mix_mask_pred = v_mask_states[1][0].detach().data.cpu()
                    v_mix_mask_nonused = (((v_mask_states[1][0] * backgrounds[0])+1)*0.5).detach().data.cpu()

                    # v_masks = v_mask_states[0][i_loss].clone().detach().cpu().detach()
                    traj_pred = m_predictions[i_loss].detach().cpu().detach()
                    traj_target = m_targets[i_loss].detach().cpu().detach()



                    optimization_results[indices_local_rep[i_loss]]['min_vis']={'pred': vision_predictions.numpy(),
                                                                          'cv_pred': cv_prediction.numpy(),
                                                                          'target': vision_targets.numpy(),
                                                                          'l0mem': l0mem,
                                                                          'mem': bg.numpy(),
                                                                          'traj_pred': traj_pred.numpy(),
                                                                          'traj_target': traj_target.numpy(),
                                                                          'm_cv_states': m_cv_state,
                                                                          'v_states': v_state,
                                                                          'l0mem_selection': l0mem_selection,
                                                                          'l0mem_maskupscale': l0mem_maskupscale,
                                                                          'l0mem_updatemask': l0mem_updatemask,
                                                                          'l0memupdatemask_pred': l0memupdatemask_pred,
                                                                          'canvas_': canvas_,
                                                                          'maskedl0mem': maskedl0mem,
                                                                          'wnds': wnds,
                                                                          'l0mem_updatesource_': l0mem_updatesource_,
                                                                          'l0mem_outmix': l0mem_outmix,
                                                                          'l0mem_memmix': l0mem_memmix,
                                                                          'v_mix_mask_refresh': v_mix_mask_refresh.numpy(),
                                                                          'v_mix_mask_pred': v_mix_mask_pred.numpy(),
                                                                          'v_mix_mask_nonused': v_mix_mask_nonused.numpy(),
                                                                          }

                    if args.model_args['language_args']["is_lang"]:
                        l_prediction = l_predictions[i_loss].detach().cpu().detach()
                        l_target = l_targets[i_loss].detach().cpu().numpy()
                        optimization_results[indices_local_rep[i_loss]]['min_vis']['l_pred'] = l_prediction.numpy()
                        optimization_results[indices_local_rep[i_loss]]['min_vis']['l_targ'] =  l_target
                        if args.model_args['language_args']['is_pb']:
                            optimization_results[indices_local_rep[i_loss]]['min_vis']['l_pb'] = lang_pbs.detach().cpu().numpy()

                    optimization_results[indices_local_rep[i_loss]]['configs'].append(c_cfg)

                optimization_results[indices_local_rep[i_loss]]['losses'].append(c_loss)


            regression_optimizer['pv_mu'].zero_grad()
            regression_optimizer['pv_logvar'].zero_grad()
            if not args.model_args['integration_args']["is_UG"]:
                regression_optimizer['pvrnn_prior_i']['mu'].zero_grad()
                regression_optimizer['pvrnn_prior_i']['logvar'].zero_grad()

            regression_optimizer['lang_pb'].zero_grad()

            loss.backward()

            regression_optimizer['pv_mu'].step()
            regression_optimizer['pv_logvar'].step()
            if not args.model_args['integration_args']["is_UG"]:
                regression_optimizer['pvrnn_prior_i']['mu'].step()
                regression_optimizer['pvrnn_prior_i']['logvar'].step()

            regression_optimizer['lang_pb'].step()

            t_end = time.time()
            print('time :', t_end-t_start)
            # print("pvmu={}".format(pv_mu.detach().cpu().numpy().mean()))
            # print("pv_ar={}".format(torch.exp(pv_logvar).detach().cpu().numpy().mean()))

            if do_eval_this_iter and False:
                # save results

                model.transformer.where_on_image(attention_wheres, rv_predictions)

                vision_predictions = (rv_predictions.detach().data.cpu() + 1) * 0.5
                cv_predictions[cv_predictions == 0] = -1
                cv_predictions = (cv_predictions.detach() + 1) * 0.5
                if (model.pv_available):
                    pv_predictions = (pv_predictions.detach().data.cpu() + 1) * 0.5
                # v_targets = (v_targets.detach().data.cpu() + 1) * 0.5
                motor_predictions = m_predictions.detach().data.cpu()

                background_plots = backgrounds.clone().detach().data.cpu()
                background_plots = (background_plots+1)*0.5

                wnds = model.transformer.image_to_window(visions[0,0:-1,:,:,:], attention_wheres[0])
                px=int(wnds.size()[2] / 2)
                py=int(wnds.size()[3] / 2)
                wnds[:, 0, px:px+2, py:py+2] = 1
                if wnds.shape[1]==3:
                    wnds[:, 1, px:px + 2, py:py + 2] = -1
                    wnds[:, 2, px:px + 2, py:py + 2] = -1
                wnds_re = model.transformer.window_to_image(wnds, attention_wheres[0], None)

                prewdists = rv_predictions-  visions[:,:-1,:,:]
                v3 = prewdists[0].clone().detach().abs().cpu()
                if v3.shape[1] == 3:
                    v4 = torch.cat([v3[:,0:1,:,:] / v3[:,0:1,:,:].max(), v3[:,1:2,:,:] / v3[:,1:2,:,:].max(), v3[:,2:3,:,:] / v3[:,2:3,:,:].max()], dim=1)
                else:
                    v4 = v3[:, 0:1, :, :] / v3[:, 0:1, :, :].max()

                wnds_cv = model.transformer.image_to_window(cv_predictions[0, :, :, :], attention_wheres[0])

                v_mix_mask = v_mask_states[2][0].clone().detach().data.cpu()
                # vis.images((v_mix_mask), opts={'title': 'MixMask memory refresh' })
                v_mix_mask = v_mask_states[1][0].clone().detach().data.cpu()
                # vis.images((v_mix_mask), opts={'title': 'MixMask mem pred'})
                v_mix_mask = (v_mask_states[1][0]*backgrounds[0]).clone().detach().data.cpu()
                # vis.images(((v_mix_mask + 1) * 0.5), opts={'title': 'non used mem'})

                background = backgrounds[0].clone().detach().data.cpu()
                background[background > 1.0] = 1.0  # TODO bakckground > 0 ??

                # inverse TPM
                imgVis = L0Memory_states[0][0].clone().detach().data.cpu()
                # vis.images((imgVis + 1) * 0.5, opts={'title': 'L0Memory '})
                imgVis = L0Memory_states[1][0].clone().detach().data.cpu()
                # vis.images(imgVis, opts={'title': 'L0Memory outmix' })
                imgVis = L0Memory_states[2][0].clone().detach().data.cpu()
                # vis.images(imgVis, opts={'title': 'L0Memory memmix'})

                for idx, v, cv, pv, m, mask, bg, mem_outmix, l0mem, att \
                        in itertools.zip_longest(indices, vision_predictions, cv_predictions, pv_predictions,
                                                 motor_predictions, mask_targets, background_plots, v_mask_states[1],
                                                 L0Memory_states[0], attention_wheres, fillvalue=[]):
                    # save vision prediction as a gif
                    length = int(mask.sum().data.item())
                    masked_v = v[:] #torch.masked_select(v, mask)
                    masked_cv = cv[:]
                    masked_pv = pv[:]
                    masked_m = m[:]

                    io.save_mot(m, 'motor_{}_{:04d}'.format(step, idx))
                    images = masked_v #[transforms.ToPILImage()(image) for image in masked_v]
                    io.save_gif(images[::2][:40], 'vision_{}_{:04d}'.format(step, idx))
                    images = masked_cv #[transforms.ToPILImage()(image) for image in masked_cv]
                    io.save_gif(images[::2][:40], 'cv_{}_{:04d}'.format(step, idx))

                    if (model.pv_available):
                        images = masked_pv #[transforms.ToPILImage()(image) for image in masked_pv]
                        io.save_gif(images[::2][:40], 'pv_{}_{:04d}'.format(step, idx))

                    io.save_gif(bg[::2][:40], 'memory_{}_{:04d}'.format(step, idx))
                    wnds = model.transformer.image_to_window(mem_outmix[:], att[:])

                    maskedl0mem = ((l0mem + 1) * 0.5)*wnds
                    io.save_gif(maskedl0mem[:10], 'maskedl0mem_{}_{:04d}'.format(step, idx))

                    unmaskedl0mem = ((l0mem + 1) * 0.5)
                    io.save_gif(unmaskedl0mem[::2][:40], 'unmaskedl0mem_{}_{:04d}'.format(step, idx))
                    io.save_txt(masked_m, 'motor_{}_{:04d}'.format(step, idx))

                    # import pdb
                    # pdb.set_trace() b

        #plot best results:
        for i in range(len(indices)):
            basefolder = work_dir + foldernames[i]

            filename = basefolder + '/results.pk'
            with open(filename, 'wb') as f:
                pickle.dump(optimization_results[i], f)
            f.close()

            print("index: ", i)
            vis_pred = torch.tensor(optimization_results[i]['min_vis']['pred'])
            cv_pred = torch.tensor(optimization_results[i]['min_vis']['cv_pred'])
            # pv_pred = torch.tensor(optimization_results[i]['min_vis']['pv_pred'])
            m_pred = torch.tensor(optimization_results[i]['min_vis']['traj_pred'])

            if args.model_args['language_args']['is_lang']:
                l_pred = torch.tensor(optimization_results[i]['min_vis']['l_pred'])
                l_target = torch.tensor(optimization_results[i]['min_vis']['l_targ'])
                l_str_pred = io.gen_lang(l_pred)
                l_str_target = io.gen_lang(l_target)
                title = "pred = {}, target = {}".format(l_str_pred, l_str_target)
            else:
                title = "motor"
            v_states = optimization_results[i]['min_vis']['v_states']
            l0mem_selection = optimization_results[i]['min_vis']['l0mem_selection']
            l0mem_maskupscale = optimization_results[i]['min_vis']['l0mem_maskupscale']
            l0mem_updatemask = optimization_results[i]['min_vis']['l0mem_updatemask']
            canvas_ = optimization_results[i]['min_vis']['canvas_']
            l0memupdatemask_pred = optimization_results[i]['min_vis']['l0memupdatemask_pred']
            m_cv_states = optimization_results[i]['min_vis']['m_cv_states']
            m_target = torch.tensor(optimization_results[i]['min_vis']['traj_target'])

            vis_target = torch.tensor(optimization_results[i]['min_vis']['target'])
            vis_l0 = optimization_results[i]['min_vis']['l0mem']
            # print("visl0shape={}".format(vis_l0.shape))
            if vis_l0 is not None:
                l0mem_updatesource = optimization_results[i]['min_vis']['l0mem_updatesource_']

                wnds = torch.tensor(optimization_results[i]['min_vis']['wnds'])

                v_mix_mask_refresh = torch.tensor(optimization_results[i]['min_vis']['v_mix_mask_refresh'])
                v_mix_mask_nonused = None if type(optimization_results[i]['min_vis']['v_mix_mask_nonused']) == None else torch.tensor(
                    optimization_results[i]['min_vis']['v_mix_mask_nonused'])
                l0mem_outmix = None if type(optimization_results[i]['min_vis']['l0mem_outmix']) == None else torch.tensor(
                    optimization_results[i]['min_vis']['l0mem_outmix'])
                l0mem_memmix = None if type(optimization_results[i]['min_vis']['l0mem_memmix']) == None else torch.tensor(
                    optimization_results[i]['min_vis']['l0mem_memmix'])
                v_mix_mask_pred = None if type(optimization_results[i]['min_vis']['v_mix_mask_pred']) == None else torch.tensor(
                    optimization_results[i]['min_vis']['v_mix_mask_pred'])
                maskedl0mem = None if type(optimization_results[i]['min_vis']['maskedl0mem']) == None else torch.tensor(
                    optimization_results[i]['min_vis']['maskedl0mem'])
                vis_l0 = torch.tensor(vis_l0)

            vis_mem = torch.tensor(optimization_results[i]['min_vis']['mem'])
            vis_diff = vis_target - vis_pred
            f = 5 # plot every fth frame
            softmax_config = np.load("softmax_config.npy", allow_pickle=True)
            io.save_mot(m_pred,  foldernames[i]+'/m_pred')
            io.save_mot(m_target,  foldernames[i]+'/m_target')
            b_state_dict = {
                        "v": v_states, "m_cv": m_cv_states,
                        "mu_p": mu_p, "sigma_p": sigma_p, "mu_q": mu_q, "sigma_q": sigma_q,
                        "m_pred":m_pred, "m_targ": m_target}  #integ_states': ib_state,
            if args.model_args['language_args']['is_lang']:
                b_state_dict['l_pred'] = l_pred
                b_state_dict['l_targ'] = l_target
            io.save_b_state(b_state_dict, foldernames[i]+'/b_state')
            io.save_gif(vis_pred[::f], foldernames[i]+'/pred')
            io.save_gif(vis_target[::f], foldernames[i]+'/target')
            io.save_gif(cv_pred[::f], foldernames[i] + '/cv_pred')
            io.save_gif((vis_mem[::f] + 1) * 0.5, foldernames[i] + '/l0mem')

            if vis_l0 is None:
                vis1 = torch.cat((vis_mem, cv_pred), 2)
                vis2 = torch.cat((vis1, vis_pred), 2)
                vis3 = torch.cat((vis2, vis_target), 2)
                vis4 = torch.cat((vis3, vis_diff), 2)
                io.save_video((vis4 + 1) * 0.5, foldernames[i] + '/vispred',
                              figsize=(0.01 * vis4.shape[2], 0.1 * vis4.shape[3]))

            if vis_l0 is not None:
                io.save_gif(l0mem_maskupscale[::f], foldernames[i] + '/l0mem_maskupscale')
                io.save_gif(l0mem_selection[::f], foldernames[i] + '/l0mem_selection')
                io.save_gif(l0mem_updatemask[::f], foldernames[i] + '/l0mem_updatemask')
                io.save_gif(wnds[::f], foldernames[i] + '/wnds')
                io.save_gif(l0memupdatemask_pred[::f], foldernames[i] + '/l0memupdatemask_pred')
                io.save_gif(canvas_[::f], foldernames[i] + '/canvas_')
                io.save_gif(v_mix_mask_refresh[::f], foldernames[i] + '/v_mix_mask_refresh')

                io.save_gif((vis_l0[::f]+1)*0.5, foldernames[i]+'/l1')
                io.save_gif((l0mem_memmix[::f] + 1) * 0.5, foldernames[i] + '/l0memmix')
                io.save_gif((l0mem_outmix[::f] + 1) * 0.5, foldernames[i] + '/l0outmix')
                io.save_gif((v_mix_mask_pred[::f] + 1) * 0.5, foldernames[i] + '/vmixmask')
                io.save_gif((v_mix_mask_nonused[::f] + 1) * 0.5, foldernames[i] + '/vmixmask_nonused')
                io.save_gif((l0mem_updatesource[::f] + 1) * 0.5, foldernames[i] + '/l0mem_updatesource')
                io.save_gif((maskedl0mem[::f] + 1) * 0.5, foldernames[i] + '/l1masked')

                l1memmasked = []
                for j in range(len(maskedl0mem)):
                    new_frame = np.moveaxis(maskedl0mem[j].detach().cpu().numpy(), 0, -1)
                    frame = cv2.resize(new_frame, (64, 64))
                    frame = np.rollaxis(frame, 2, 0)
                    l1memmasked.append(frame)
                l1memmasked = torch.from_numpy(np.array(l1memmasked))
                # c_im1 = torch.cat((l1memmasked, vis_mem), 2)
                # c_im2 = torch.cat((c_im1, cv_pred), 2)
                # c_im3 = torch.cat((c_im2, vis_pred), 2)
                # c_im4 = torch.cat((c_im3, vis_target), 2)
                # c_im1 = torch.cat((vis_l0, maskedl0mem), 2)
                # c_im2 = torch.cat((vis_pred, vis_target), 2)
                io.save_gif((vis_mem[::f]+1)*0.5, foldernames[i]+'/l0mem')

                vis1 = torch.cat((l1memmasked, vis_mem), 2)
                vis2 = torch.cat((vis1, cv_pred), 2)
                vis3 = torch.cat((vis2, vis_pred), 2)
                vis4 = torch.cat((vis3, vis_target), 2)
                vis5 = torch.cat((vis4, vis_diff), 2)
                io.save_gif(vis5[::f], foldernames[i] + '/vis_')
                # io.save_gif(v_mask, foldernames[i]+'/mask')
            # io.save_video(vis_pred, foldernames[i]+'/pred')
            # io.save_video(vis_target, foldernames[i]+'/target')
            if vis_l0 is not None:
                # io.save_video((vis_l0+1)*0.5, foldernames[i]+'/l1')
                # io.save_video((c_im1+1)*0.5, foldernames[i]+'/l1memmix')
                # io.save_video((maskedl0mem+1)*0.5, foldernames[i]+'/l1masked')
                io.save_video((vis5+1)*0.5, foldernames[i]+'/vis', figsize=(0.01*vis5.shape[2], 0.1*vis5.shape[3]))
            # io.plot_latentstates(int_states, name=foldernames[i]+"/pvrnn_d")
            mot_diff = io.save_motor(m_pred, m_target, foldernames[i] + "/motor", softmax_config=softmax_config, title=title)
            print(title)
            # io.save_video((vis_mem+1)*0.5, foldernames[i]+'/l0mem')
        foldernames
        #final execution ?!


        intention = {'mu': torch.stack([target_intention_params[i]['mu'] for i in indices], dim=0),
                     'logvar': torch.stack([target_intention_params[i]['logvar'] for i in indices], dim=0)}

        intention_mu = intention['mu'].to(dev)
        intention_logvar = intention['logvar'].to(dev)

        vis_diff = F.mse_loss(vis_pred, vis_target).mean()

        diff = mot_diff + vis_diff

    return lang_labels, lang_latent

def start():
    io.print_log('Parameters:\n{}\n'.format(str(vars(args))))
    id=str(uuid.uuid4())
    # test phase
    trianstates = eval(args.trainstates)
    vver = '/v'+str(evalversion)
    try:
        os.makedirs(work_dir+vver)
    except FileExistsError:
        None
    reps = 1
    for rep in range(reps):
        # need to fix loading trained model parameters
        for trainstate in trianstates:
            # load checkpoint:
            print(model)
            filename = 'epoch{}_checkpoint.pt'.format(trainstate)
            checkpoint = io.load_checkpoint(filename, dev1=dev)

            try:
                os.makedirs(work_dir + vver)
            except FileExistsError:
                None
                prefix = vver + '/trainstate' + str(trainstate) + '/rep' + str(rep) + '_' + id + "/"
            try:
                os.makedirs(work_dir + prefix)
            except FileExistsError:
                None
            io.load_weights_from_checkpoint(model, checkpoint)

            with open(work_dir+prefix + 'notes.txt', 'x') as f:
                f.write("notes on evaluation \n")

                posterior_mu_train = io.load_from_checkpoint(checkpoint, param='pvrnn_pos_mu')
                posterior_logvar_train = io.load_from_checkpoint(checkpoint, param='pvrnn_pos_logvar')
                if args.model_args['integration_args']['is_UG'] == False:
                    pvrnn_prior_mu, pvrnn_prior_logvar = \
                        io.load_from_checkpoint(checkpoint, param='pvrnn_prior_mu_i'), \
                            io.load_from_checkpoint(checkpoint, param="pvrnn_prior_logvar_i")
                else:
                    pvrnn_prior_mu, pvrnn_prior_logvar = pvrnn_prior_mu_i, pvrnn_prior_logvar_i
                pvrnn_prior = {'mu': pvrnn_prior_mu, 'logvar': pvrnn_prior_logvar}
                # else:
                #     prior = io.load_from_checkpoint(checkpoint, param='prior')
                if args.model_args['language_args']['is_lang']:
                    # if args.model_args['language_args']['is_pb']:
                    lang_pb_train = io.load_from_checkpoint(checkpoint, param='lang_pb')
                    lang_pb_train_ = [lang_pb_train[i].view(1, -1) for i in range(len(lang_pb_train))]
                    lang_pb_train = torch.cat(lang_pb_train_).detach().cpu()
                    lang_train_labels = io.load_from_checkpoint(checkpoint, param='lang_train_labels')
                    lang_train_labels = torch.cat(lang_train_labels).detach().cpu()

                    lang_pb = lang_pb_test
                # evaluation
                io.print_log('Evaluation Start:')
                if args.test_feeder_args['selectTrain']:
                    f.write("using training data for evaluation \n")
                    lang_pb_load = io.load_from_checkpoint(checkpoint, param='lang_pb')

                    pv_pos_mu, pv_pos_logvar = io.load_from_checkpoint(checkpoint, param='pvrnn_pos_mu'), io.load_from_checkpoint(checkpoint, param='pv_pos_logvar')
                    pv_posterior_load = {'pv_pos_mu': pv_pos_mu, 'pv_pos_logvar': pv_pos_logvar}
                    lang_labels, lang_latent = test(pv_posterior=pv_posterior_load, pv_prior=pvrnn_prior,
                                                    lang_pb=lang_pb_load, prefix=prefix,
                                                    pred_lang=False, vision_goal=False)
                elif args.model_args['language_args']['is_lang']:
                    pred_lang = False
                    vision_goal = False

                    fn = "lang_latent_{}_predlang_{}".format(trainstate, pred_lang)
                    f.write(" predicting language = {}\n vision goal = {}".format(pred_lang, vision_goal))
                    lang_labels, lang_latent = test(pv_prior=pvrnn_prior,
                                                                     lang_pb=lang_pb, prefix=prefix,
                                                                     pred_lang=pred_lang, vision_goal=vision_goal)
                elif args.model_args['integration_args']['is_pvrnn']:
                    print("pvrnn + no language")
                    lang_labels, lang_latent= test(pv_prior=pvrnn_prior, prefix=prefix,
                                                                     vision_goal=True)
                else:
                    print("no language")
                    lang_labels, lang_latent= test(prefix=prefix, vision_goal=True)
                io.print_log('Done.\n')
                if args.model_args['language_args']['is_lang']:
                    if args.model_args['language_args']['is_pb']:
                        lang_labels = torch.cat(lang_labels)
                        lang_latent = [lang_latent[i].view(1, -1) for i in range(len(lang_latent))]
                        lang_latent = torch.cat(lang_latent).detach().cpu()
                        lang_latent_ = torch.cat((lang_pb_train, lang_latent))
                        lang_labels_ = torch.cat((lang_train_labels, lang_labels))

                    io.plot_lang_latent(dir=work_dir+prefix, lang_states=lang_latent_.numpy(), labels=lang_labels_.numpy(), colors=['red', 'green', 'blue', 'purple', 'yellow'], fn=fn)
            f.close()


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    start()



