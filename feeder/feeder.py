# sys
# import os
# import sys
# import numpy as np
# import random
# import pickle
import h5py
import numpy as np
# torch
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.utils.rnn import pad_sequence

# class Feeder(torch.utils.data.Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path

#         self.load_data()

#     def load_data(self):
#         self.data = h5py.File(self.data_path, 'r')
#     	# if mmap:
#      #    	data = np.load(self.data_path, mmap_mode='r')
#      #    else:
#      #    	data = np.load(self.data_path)

#     	# self.vision_data = data['vision']
#     	# self.motor_data = data['motor']
#     	# self.length_data = data['']



#     def __len__(self):
#         assert self.data['vision'].shape[0] == self.data['motor'].shape[0] == self.data['mask'].shape[0], 'number of samples are not matched'

#         return self.data['vision'].shape[0]


#     def __getitem__(self, index):
#     	# get data
#         vision = self.data['vision'][index]
#         motor = self.data['motor'][index]
#         mask = self.data['mask'][index]
        

#         return index, vision, motor, mask
import random




class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, selectTrain=False, selectTest=False, runnr=0, useMotor=False):
        self.data_path = data_path
        # self.num_context_frames = num_context_frames
        self.load_data()

    def load_data(self):
        self.data = h5py.File(self.data_path, 'r')

        self.dataThreadsaveVision = self.data['vision'][:]
        self.dataThreadsaveMotor = self.data['motor'][:]
        self.dataThreadsaveMask = self.data['mask'][:]
        self.dataThreadsaveLanguage = self.data['language'][:]
        self.dataThreadsaveLanguageMask = self.data['lang_mask'][:]

        # self.vision_data = np.load('./data/atn_vision_200.npy')#, mmap_mode='r')
        # self.motor_data = np.load('./data/atn_motor_softmax_200_prob_q10.npy')#, mmap_mode='r')
        # if mmap:
     #      data = np.load(self.data_path, mmap_mode='r')
     #    else:
     #      data = np.load(self.data_path)
        # self.vision_data = data['vision']
        # self.motor_data = data['motor']
        # self.length_data = data['']

    def __len__(self):
        # assert self.data['vision'].shape[0] == self.data['motor'].shape[0] == self.data['mask'].shape[0], 'number of samples are not matched'

        return self.data['vision'].shape[0]

    # def __getitem_permutations__(self, index):
    #
    #     visions=[self.dataThreadsaveVision[index].copy()]
    #
    #
    #     for i in range(3):
    #             nc = i
    #             if nc==0:
    #                idx1=0
    #                idx2=1
    #             elif nc==1:
    #                idx1=1
    #                idx2=2
    #             else:
    #                idx1=0
    #                idx2=2
    #
    #
    #
    #
    #             for j in range(2):
    #                 vision = self.dataThreadsaveVision[index].copy()
    #                 tmp = vision[:, idx1, :, :].copy()
    #
    #                 vision[:, idx1, :, :] = vision[:, idx2, :, :]
    #                 vision[:, idx2, :, :] = tmp
    #
    #                 if j>0:
    #                     idx1 = 1
    #                     idx2 = 2
    #                     tmp = vision[:, idx1, :, :].copy()
    #
    #                     vision[:, idx1, :, :] = vision[:, idx2, :, :]
    #                     vision[:, idx2, :, :] = tmp
    #
    #
    #                 visions.append(vision)
    #
    #     motor = self.dataThreadsaveMotor[index]
    #     mask = self.dataThreadsaveMask[index]
    #
    #     #flipping color layers
    #
    #     return index, visions, motor, mask  # , intention

    # def __getitem__(self, index):
    #
    #     vision = self.dataThreadsaveVision[index]
    #     motor = self.dataThreadsaveMotor[index]
    #     mask = self.dataThreadsaveMask[index]
    #     # print(vision.shape)
    #     # vision = np.concatenate((np.repeat(vision[None, 0], self.num_context_frames-1, axis=0), vision))
    #     # motor = np.concatenate((np.repeat(motor[None, 0], self.num_context_frames-1, axis=0), motor))
    #     # mask = np.concatenate((np.repeat(mask[None, 0], self.num_context_frames-1, axis=0), mask))
    #     # print(vision.shape)
    #     # intention = self.data['intention'][index]
    #
    #     # get data
    #     # vision = self.vision_data[index].transpose((0,3,1,2))
    #     # motor = self.motor_data[index]
    #
    #     # mask = vision.reshape(vision.shape[0], -1).sum(1)
    #     # mask = 1 * (mask > 0)
    #
    #     return index, vision, motor, mask  # , intention
    def __getitem_permutations__(self, index):

        visions=[self.dataThreadsaveVision[index].copy()]


        for i in range(3):
                nc = i
                if nc==0:
                   idx1=0
                   idx2=1
                elif nc==1:
                   idx1=1
                   idx2=2
                else:
                   idx1=0
                   idx2=2

                for j in range(2):
                    vision = self.dataThreadsaveVision[index].copy()
                    tmp = vision[:, idx1, :, :].copy()

                    vision[:, idx1, :, :] = vision[:, idx2, :, :]
                    vision[:, idx2, :, :] = tmp

                    if j>0:
                        idx1 = 1
                        idx2 = 2
                        tmp = vision[:, idx1, :, :].copy()

                        vision[:, idx1, :, :] = vision[:, idx2, :, :]
                        vision[:, idx2, :, :] = tmp

                    visions.append(vision)

        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]
        language = self.dataThreadsaveLanguage[index]
        lang_masks = self.dataThreadsaveLanguageMask[index]
        #flipping color layers

        return index, visions, motor, language, mask, lang_masks

    def __getitem__(self, index):

        vision = self.dataThreadsaveVision[index]
        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]
        language = self.dataThreadsaveLanguage[index]
        lang_masks = self.dataThreadsaveLanguageMask[index]

        # print(vision.shape)
        # vision = np.concatenate((np.repeat(vision[None, 0], self.num_context_frames-1, axis=0), vision))
        # motor = np.concatenate((np.repeat(motor[None, 0], self.num_context_frames-1, axis=0), motor))
        # mask = np.concatenate((np.repeat(mask[None, 0], self.num_context_frames-1, axis=0), mask))
        # print(vision.shape)
        # intention = self.data['intention'][index]

        # get data
        # vision = self.vision_data[index].transpose((0,3,1,2))
        # motor = self.motor_data[index]

        # mask = vision.reshape(vision.shape[0], -1).sum(1)
        # mask = 1 * (mask > 0)

        return index, vision, motor, language, mask, lang_masks

# def collate(batch):
#     visions = [torch.from_numpy(b) for b in batch]
#     visions = pad_sequence(visions)

#     motors = [torch.from_numpy(b) for b in batch]
#     motors = pad_sequence(motors)


#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]


class FeederToyData(torch.utils.data.Dataset):
    def __init__(self, data_path, selectTrain=False, selectTest=False, runnr=0, useMotor=False):
        self.data_path = data_path
        # self.num_context_frames = num_context_frames

        self.selectTrain = selectTrain
        self.selectTest = selectTest
        self.runnr=runnr
        self.useMotor=useMotor

        self.load_data()

    def load_data(self):
        self.data = h5py.File(self.data_path, 'r')

        self.dataThreadsaveVision = self.data['vision'][:]
        self.dataThreadsaveMotor = self.data['motor'][:]
        self.dataThreadsaveLanguage = self.data['language'][:]




        alllen = self.data['vision'].shape[0]




        if self.selectTrain and (self.selectTest == False):
            self.selection_idxs = self.data['trainsamples_idx'][self.runnr,:].astype(int)


        elif (self.selectTrain == False) and self.selectTest:
            self.selection_idxs = [x for x in range(alllen) if x not in self.data['trainsamples_idx'][self.runnr, :]]
        else:
            self.selection_idxs = list(range(alllen))


        self.mask = np.ones((len(self.selection_idxs), self.data['vision'].shape[1]))

        # self.vision_data = np.load('./data/atn_vision_200.npy')#, mmap_mode='r')
        # self.motor_data = np.load('./data/atn_motor_softmax_200_prob_q10.npy')#, mmap_mode='r')

        # if mmap:

    #      data = np.load(self.data_path, mmap_mode='r')
    #    else:
    #      data = np.load(self.data_path)

    # self.vision_data = data['vision']
    # self.motor_data = data['motor']
    # self.length_data = data['']

    def __len__(self):
        # assert self.data['vision'].shape[0] == self.data['motor'].shape[0] == self.data['mask'].shape[0], 'number of samples are not matched'

        return len(self.selection_idxs) #self.data['vision'].shape[0]

    def __getitem__(self, index):
        vision = self.dataThreadsaveVision[self.selection_idxs[index]]
        language = self.dataThreadsaveLanguage[self.selection_idxs[index]]

        if self.useMotor:
            #motor = self.data['motor'][self.selection_idxs[index]]
            #todo motor data is rescaled according to vision size!!
            motor = self.dataThreadsaveMotor[self.selection_idxs[index]] / vision.shape[2:4]
        else:
            motor = [] #
        mask = self.mask[index]
        # print(vision.shape)
        # vision = np.concatenate((np.repeat(vision[None, 0], self.num_context_frames-1, axis=0), vision))
        # motor = np.concatenate((np.repeat(motor[None, 0], self.num_context_frames-1, axis=0), motor))
        # mask = np.concatenate((np.repeat(mask[None, 0], self.num_context_frames-1, axis=0), mask))
        # print(vision.shape)
        # intention = self.data['intention'][index]

        # get data
        # vision = self.vision_data[index].transpose((0,3,1,2))
        # motor = self.motor_data[index]

        # mask = vision.reshape(vision.shape[0], -1).sum(1)
        # mask = 1 * (mask > 0)

        return index, vision, motor, mask, language   # , intention

# def collate(batch):
#     visions = [torch.from_numpy(b) for b in batch]
#     visions = pad_sequence(visions)

#     motors = [torch.from_numpy(b) for b in batch]
#     motors = pad_sequence(motors)


#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]

class FeederColorFlip(torch.utils.data.Dataset):
    def __init__(self, data_path, selectTrain=False, selectTest=False, runnr=0, useMotor=False):
        self.data_path = data_path
        # self.num_context_frames = num_context_frames

        self.load_data()

    def load_data(self):
        self.data = h5py.File(self.data_path, 'r')

        self.dataThreadsaveVision = self.data['vision'][:]
        self.dataThreadsaveMotor = self.data['motor'][:]
        self.dataThreadsaveMask = self.data['mask'][:]

        # self.vision_data = np.load('./data/atn_vision_200.npy')#, mmap_mode='r')
        # self.motor_data = np.load('./data/atn_motor_softmax_200_prob_q10.npy')#, mmap_mode='r')

        # if mmap:

    #      data = np.load(self.data_path, mmap_mode='r')
    #    else:
    #      data = np.load(self.data_path)

    # self.vision_data = data['vision']
    # self.motor_data = data['motor']
    # self.length_data = data['']

    def __len__(self):
        # assert self.data['vision'].shape[0] == self.data['motor'].shape[0] == self.data['mask'].shape[0], 'number of samples are not matched'

        return self.data['vision'].shape[0]

    def __getitem__(self, index):
        vision = self.dataThreadsaveVision[index]

        nc = random.randrange(0,4)
        nc2 = random.randrange(0, 2)

        idx3=-1

        if nc==0:
           idx1=0
           idx2=1
        elif nc==1:
           idx1=1
           idx2=2
        elif nc==2:
           idx1=0
           idx2=2
        elif nc==3:
           idx1=0
           idx2=1
           idx3=2
        else:
           idx1 = 0
           idx2 = 2
           idx3=1



        # if nc == 0:
        #     if nc2==0:
        #         idx1=0
        #         idx2=1
        #     else:
        #         idx1=0
        #         idx2=2
        # if nc == 1:
        #     if nc2 == 0:
        #         idx1=1
        #         idx2=0
        #     else:
        #         idx1=1
        #         idx2=2
        # else:
        #     if nc2 == 0:
        #         idx1=2
        #         idx2=0
        #     else:
        #         idx1=2
        #         idx2=1




        #idx1=random.randrange(0,3)
        #idx2=random.randrange(0,3)

        tmp = vision[:, idx1, :, :].copy()

        vision[:, idx1, :, :]=vision[:, idx2, :, :]
        vision[:, idx2, :, :]=tmp

        if idx3>0:
            tmp = vision[:, idx3, :, :].copy()
            vision[:, idx3, :, :] = vision[:, 0, :, :]
            vision[:, 0, :, :] = tmp

        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]

        #flipping color layers

        return index, vision, motor, mask  # , intention


    def __getitem_permutations__(self, index):

        visions=[self.dataThreadsaveVision[index].copy()]


        for i in range(3):
                nc = i
                if nc==0:
                   idx1=0
                   idx2=1
                elif nc==1:
                   idx1=1
                   idx2=2
                else:
                   idx1=0
                   idx2=2




                for j in range(2):
                    vision = self.dataThreadsaveVision[index].copy()
                    tmp = vision[:, idx1, :, :].copy()

                    vision[:, idx1, :, :] = vision[:, idx2, :, :]
                    vision[:, idx2, :, :] = tmp

                    if j>0:
                        idx1 = 1
                        idx2 = 2
                        tmp = vision[:, idx1, :, :].copy()

                        vision[:, idx1, :, :] = vision[:, idx2, :, :]
                        vision[:, idx2, :, :] = tmp


                    visions.append(vision)

        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]

        #flipping color layers

        return index, visions, motor, mask  # , intention


class FeederColorFlipRepeatable(torch.utils.data.Dataset):
    def __init__(self, data_path, selectTrain=False, selectTest=False, runnr=0, useMotor=False):
        self.data_path = data_path
        # self.num_context_frames = num_context_frames

        self.rnd = random.Random()
        self.rnd.seed(0)

        self.load_data()

    def load_data(self):
        self.data = h5py.File(self.data_path, 'r')

        self.dataThreadsaveVision = self.data['vision'][:]
        self.dataThreadsaveMotor = self.data['motor'][:]
        self.dataThreadsaveMask = self.data['mask'][:]

        # self.vision_data = np.load('./data/atn_vision_200.npy')#, mmap_mode='r')
        # self.motor_data = np.load('./data/atn_motor_softmax_200_prob_q10.npy')#, mmap_mode='r')

        # if mmap:

    #      data = np.load(self.data_path, mmap_mode='r')
    #    else:
    #      data = np.load(self.data_path)

    # self.vision_data = data['vision']
    # self.motor_data = data['motor']
    # self.length_data = data['']

    def __len__(self):
        # assert self.data['vision'].shape[0] == self.data['motor'].shape[0] == self.data['mask'].shape[0], 'number of samples are not matched'

        return self.data['vision'].shape[0]

    def __getitem__(self, index):
        vision = self.dataThreadsaveVision[index]

        nc = self.rnd.randrange(0,4)
        nc2 = self.rnd.randrange(0, 2)

        idx3=-1

        if nc==0:
           idx1=0
           idx2=1
        elif nc==1:
           idx1=1
           idx2=2
        elif nc==2:
           idx1=0
           idx2=2
        elif nc==3:
           idx1=0
           idx2=1
           idx3=2
        else:
           idx1 = 0
           idx2 = 2
           idx3=1



        # if nc == 0:
        #     if nc2==0:
        #         idx1=0
        #         idx2=1
        #     else:
        #         idx1=0
        #         idx2=2
        # if nc == 1:
        #     if nc2 == 0:
        #         idx1=1
        #         idx2=0
        #     else:
        #         idx1=1
        #         idx2=2
        # else:
        #     if nc2 == 0:
        #         idx1=2
        #         idx2=0
        #     else:
        #         idx1=2
        #         idx2=1




        #idx1=random.randrange(0,3)
        #idx2=random.randrange(0,3)

        tmp = vision[:, idx1, :, :].copy()

        vision[:, idx1, :, :]=vision[:, idx2, :, :]
        vision[:, idx2, :, :]=tmp

        if idx3>0:
            tmp = vision[:, idx3, :, :].copy()
            vision[:, idx3, :, :] = vision[:, 0, :, :]
            vision[:, 0, :, :] = tmp

        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]

        #flipping color layers

        return index, vision, motor, mask  # , intention


    def __getitem_permutations__(self, index):

        visions=[self.dataThreadsaveVision[index].copy()]


        for i in range(3):
                nc = i
                if nc==0:
                   idx1=0
                   idx2=1
                elif nc==1:
                   idx1=1
                   idx2=2
                else:
                   idx1=0
                   idx2=2




                for j in range(2):
                    vision = self.dataThreadsaveVision[index].copy()
                    tmp = vision[:, idx1, :, :].copy()

                    vision[:, idx1, :, :] = vision[:, idx2, :, :]
                    vision[:, idx2, :, :] = tmp

                    if j>0:
                        idx1 = 1
                        idx2 = 2
                        tmp = vision[:, idx1, :, :].copy()

                        vision[:, idx1, :, :] = vision[:, idx2, :, :]
                        vision[:, idx2, :, :] = tmp


                    visions.append(vision)

        motor = self.dataThreadsaveMotor[index]
        mask = self.dataThreadsaveMask[index]

        #flipping color layers

        return index, visions, motor, mask  # , intention



