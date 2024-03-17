#!/usr/bin/env python
import argparse
import os
import sys
import traceback
import time
import warnings
import pickle
from collections import OrderedDict
import yaml
import numpy as np
# torch
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
#videovis:
#import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import sklearn.cluster as cluster
from umap import UMAP
from PIL import Image
import scipy.stats as stats
import seaborn as sns
from data_processing import *
dp = data_processing()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
class IO():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.pavi_logger = None
        self.session_file = None
        self.model_text = ''
        
    # PaviLogger is removed in this version
    def log(self, *args, **kwargs):
        pass
    #     try:
    #         if self.pavi_logger is None:
    #             from torchpack.runner.hooks import PaviLogger
    #             url = 'http://pavi.parrotsdnn.org/log'
    #             with open(self.session_file, 'r') as f:
    #                 info = dict(
    #                     session_file=self.session_file,
    #                     session_text=f.read(),
    #                     model_text=self.model_text)
    #             self.pavi_logger = PaviLogger(url)
    #             self.pavi_logger.connect(self.work_dir, info=info)
    #         self.pavi_logger.log(*args, **kwargs)
    #     except:  #pylint: disable=W0702
    #         pass
    def pca(self, nc):
        pca = PCA(n_components=nc)
        return pca
    def kpca(self, nc, kernel='linear'):
        kpca = KernelPCA(n_components=nc, kernel=kernel)
        return kpca
    def tsne(self, nc):
        tsne_ = TSNE(n_components=nc, random_state=0)
        return tsne_
    def isomap(self, nc):
        iso = Isomap(n_components=nc)
        return iso

    def umap(self, nc, n_neighbors=40, metric='cosine', min_dist=0.001, local_connectivity=0):
        umap_ = UMAP(n_neighbors=n_neighbors , n_components=nc, metric=metric, min_dist=min_dist, local_connectivity=local_connectivity)
        return umap_
    def pca_states(self, states, nc=5):
        """
        states of one layer of the neural network
        timesteps, batchsize, layer, [h,c] (we want only h) , D (dim)
        """
        len_ = len(states[0])
        pca_ = self.pca(nc)
        new_states = []
        for l in range(len(states[0])):
            for t in range(len(states)):
                new_states.append(states[t][l][0].detach().cpu().numpy())
        new_states = np.array(new_states)
        states_pca = []
        for l in range(len(new_states[0])):
            pcas = pca_.fit_transform(new_states[:, l, :])
            states_pca.append(pcas)
        return new_states, states_pca

    def cluster_latent(self, n):
        return cluster.KMeans(n_clusters=n)

    def correlation_matrix(self, data):
        corr_matrix = np.corrcoef(data).round(decimals=2)
        return corr_matrix

    def dm_red(self,data, method='kpca', params={}):
        """
        apply specified dimensionlity reduction to data
        """
        reduced_data=[]

        return reduced_data

    def normalize(self, vec):
        l = len(vec)
        max = np.max(vec)
        min = np.min(vec)
        normalized_vec = [(vec[i]-min)/(max-min) for i in range(l)]
        return normalized_vec

    def sents(self):
        nouns = ['red', 'green', 'blue', 'purple', 'yellow']
        verbs = ['grasp . . .', 'move left . .', 'move right . .', 'move front . .', 'move back . .', 'put on green .',
                 'put on blue .', 'put on yellow .']
        v_ = []
        sentences = []
        sentences_vecs = []
        for v in range(len(verbs)):
            v_.append(verbs[v].split(" "))
        for n in range(len(nouns)):
            for i in range(len(v_)):
                s = [v_[i][0], nouns[n], v_[i][1], v_[i][2], v_[i][3]]
                sent = " ".join(s)
                sentences.append(sent)
                sent_vec, _ = dp.lang_vec(sent, max_len=5)
                sentences_vecs.append(np.array(sent_vec))
        return sentences, sentences_vecs


    def get_mean_latent(self, latent_pca_1, latent_pca_2, latent_state_labels):
        grasp_red, grasp_green, grasp_blue, grasp_purple, grasp_yellow = [], [], [], [], []
        red_left, green_left, blue_left, purple_left, yellow_left = [], [], [], [], []
        red_right, green_right, blue_right, purple_right, yellow_right = [], [], [], [], []
        red_front, green_front, blue_front, purple_front, yellow_front = [], [], [], [], []
        red_back, green_back, blue_back, purple_back, yellow_back = [], [], [], [], []
        redongreen, greenongreen, blueongreen, purpleongreen, yellowongreen = [], [], [], [], []
        redonblue, greenonblue, blueonblue, purpleonblue, yellowonblue = [], [], [], [], []
        redonyellow, greenonyellow, blueonyellow, purpleonyellow, yellowonyellow = [], [], [], [], []
        latent_state_labels = latent_state_labels.numpy()
        red_v = dp.lang_vec("red", max_len=1)
        green_v = dp.lang_vec("green", max_len=1)
        blue_v = dp.lang_vec("blue", max_len=1)
        purple_v = dp.lang_vec("purple", max_len=1)
        yellow_v = dp.lang_vec("yellow", max_len=1)
        left_v = dp.lang_vec("left", max_len=1)
        right_v = dp.lang_vec("right", max_len=1)
        front_v = dp.lang_vec("front", max_len=1)
        back_v = dp.lang_vec("back", max_len=1)
        grasp_v = dp.lang_vec("grasp", max_len=1)
        put_v = dp.lang_vec("put", max_len=1)
        for i in range(len(latent_state_labels)):
            if np.argmax(latent_state_labels[i][0]) == np.argmax(grasp_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    grasp_red.append([latent_pca_1[i], latent_pca_2[i]])
                if np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    grasp_green.append([latent_pca_1[i], latent_pca_2[i]])
                if np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    grasp_blue.append([latent_pca_1[i], latent_pca_2[i]])
                if np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    grasp_purple.append([latent_pca_1[i], latent_pca_2[i]])
                if np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    grasp_yellow.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][2]) == np.argmax(left_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    red_left.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    green_left.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blue_left.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purple_left.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellow_left.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][2]) == np.argmax(right_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    red_right.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    green_right.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blue_right.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purple_right.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellow_right.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][2]) == np.argmax(front_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    red_front.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    green_front.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blue_front.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purple_front.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellow_front.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][2]) == np.argmax(back_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    red_back.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    green_back.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blue_back.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purple_back.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellow_back.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][0]) == np.argmax(put_v[0]) and np.argmax(latent_state_labels[i][3]) == np.argmax(green_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    redongreen.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    greenongreen.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blueongreen.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purpleongreen.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellowongreen.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][0]) == np.argmax(put_v[0]) and np.argmax(latent_state_labels[i][3]) == np.argmax(blue_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    redonblue.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    greenonblue.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blueonblue.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purpleonblue.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellowonblue.append([latent_pca_1[i], latent_pca_2[i]])
            elif np.argmax(latent_state_labels[i][0]) == np.argmax(put_v[0]) and np.argmax(latent_state_labels[i][3]) == np.argmax(yellow_v[0]):
                if np.argmax(latent_state_labels[i][1]) == np.argmax(red_v[0]):
                    redonyellow.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(green_v[0]):
                    greenonyellow.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(blue_v[0]):
                    blueonyellow.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(purple_v[0]):
                    purpleonyellow.append([latent_pca_1[i], latent_pca_2[i]])
                elif np.argmax(latent_state_labels[i][1]) == np.argmax(yellow_v[0]):
                    yellowonyellow.append([latent_pca_1[i], latent_pca_2[i]])


        # print(np.mean(np.array(grasp_red)[:, 0]))
        # print(np.array(grasp_red)[:, 1].shape)
        # print(np.mean(np.array(grasp_blue)[:, 0]))
        # exit()
        labels_red, labeld_green, labels_blue, labels_purple, labels_yellow= [], [], [], [], []
        grasp_red_mean, grasp_green_mean, grasp_blue_mean, grasp_purple_mean, grasp_yellow_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        red_left_mean, green_left_mean, blue_left_mean, purple_left_mean, yellow_left_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        red_right_mean, green_right_mean, blue_right_mean, purple_right_mean, yellow_right_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        red_front_mean, green_front_mean, blue_front_mean, purple_front_mean, yellow_front_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        red_back_mean, green_back_mean, blue_back_mean, purple_back_mean, yellow_back_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        redongreen_mean, greenongreen_mean, blueongreen_mean, purpleongreen_mean, yellowongreen_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        redonblue_mean, greenonblue_mean, blueonblue_mean, purpleonblue_mean, yellowonblue_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        redonyellow_mean, greenonyellow_mean, blueonyellow_mean, purpleonyellow_mean, yellowonyellow_mean = [0,0], [0,0], [0,0], [0,0], [0,0]
        if len(grasp_purple) != 0:
            grasp_red_mean, grasp_green_mean, grasp_blue_mean, grasp_purple_mean, grasp_yellow_mean =\
            [np.mean(np.array(grasp_red)[:, 0]), np.mean(np.array(grasp_red)[:, 1])], [np.mean(np.array(grasp_green)[:, 0]), np.mean(np.array(grasp_green)[:, 1])],\
            [np.mean(np.array(grasp_blue)[:, 0]), np.mean(np.array(grasp_blue)[:, 1])], [np.mean(np.array(grasp_purple)[:, 0]), np.mean(np.array(grasp_purple)[:, 1])], \
            [np.mean(np.array(grasp_yellow)[:, 0]), np.mean(np.array(grasp_yellow)[:, 1])]

        else:
            grasp_red_mean, grasp_green_mean, grasp_blue_mean, grasp_purple_mean, grasp_yellow_mean =\
            [np.mean(np.array(grasp_red)[:, 0]), np.mean(np.array(grasp_red)[:, 1])], [np.mean(np.array(grasp_green)[:, 0]), np.mean(np.array(grasp_green)[:, 1])],\
            [np.mean(np.array(grasp_blue)[:, 0]), np.mean(np.array(grasp_blue)[:, 1])], [0, 0], [0, 0]


        if len(purple_left) != 0:
            red_left_mean, green_left_mean, blue_left_mean, purple_left_mean, yellow_left_mean =\
                [np.mean(np.array(red_left)[:, 0]), np.mean(np.array(red_left)[:, 1])], [
                np.mean(np.array(green_left)[:, 0]), np.mean(np.array(green_left)[:, 1])], \
            [np.mean(np.array(blue_left)[:, 0]), np.mean(np.array(blue_left)[:, 1])], [
                np.mean(np.array(purple_left)[:, 0]), np.mean(np.array(purple_left)[:, 1])], \
            [np.mean(np.array(yellow_left)[:, 0]), np.mean(np.array(yellow_left)[:, 1])]
        elif len(red_left) !=0:
            red_left_mean, green_left_mean, blue_left_mean, purple_left_mean, yellow_left_mean =\
                [np.mean(np.array(red_left)[:, 0]), np.mean(np.array(red_left)[:, 1])], [
                np.mean(np.array(green_left)[:, 0]), np.mean(np.array(green_left)[:, 1])], \
            [np.mean(np.array(blue_left)[:, 0]), np.mean(np.array(blue_left)[:, 1])], [0,0], [0,0]

        if len(purple_right) != 0:
            red_right_mean, green_right_mean, blue_right_mean, purple_right_mean, yellow_right_mean = \
                [np.mean(np.array(red_right)[:, 0]), np.mean(np.array(red_right)[:, 1])], [
                np.mean(np.array(green_right)[:, 0]), np.mean(np.array(green_right)[:, 1])], \
            [np.mean(np.array(blue_right)[:, 0]), np.mean(np.array(blue_right)[:, 1])], [
                np.mean(np.array(purple_right)[:, 0]), np.mean(np.array(purple_right)[:, 1])], \
            [np.mean(np.array(yellow_right)[:, 0]), np.mean(np.array(yellow_right)[:, 1])]
        elif len(red_right) != 0:
            red_right_mean, green_right_mean, blue_right_mean, purple_right_mean, yellow_right_mean = \
                [np.mean(np.array(red_right)[:, 0]), np.mean(np.array(red_right)[:, 1])], [
                np.mean(np.array(green_right)[:, 0]), np.mean(np.array(green_right)[:, 1])], \
            [np.mean(np.array(blue_right)[:, 0]), np.mean(np.array(blue_right)[:, 1])], [0,0], [0,0]

        if len(purple_front) != 0:
            red_front_mean, green_front_mean, blue_front_mean, purple_front_mean, yellow_front_mean = \
                [np.mean(np.array(red_front)[:, 0]), np.mean(np.array(red_front)[:, 1])], [
                np.mean(np.array(green_front)[:, 0]), np.mean(np.array(green_front)[:, 1])], \
            [np.mean(np.array(blue_front)[:, 0]), np.mean(np.array(blue_front)[:, 1])], [
                np.mean(np.array(purple_front)[:, 0]), np.mean(np.array(purple_front)[:, 1])], \
            [np.mean(np.array(yellow_front)[:, 0]), np.mean(np.array(yellow_front)[:, 1])]
        elif len(red_front) !=0:
            red_front_mean, green_front_mean, blue_front_mean, purple_front_mean, yellow_front_mean = \
                [np.mean(np.array(red_front)[:, 0]), np.mean(np.array(red_front)[:, 1])], [
                    np.mean(np.array(green_front)[:, 0]), np.mean(np.array(green_front)[:, 1])], \
                [np.mean(np.array(blue_front)[:, 0]), np.mean(np.array(blue_front)[:, 1])], [0,0], [0,0]

        if len(purple_back) != 0:
            red_back_mean, green_back_mean, blue_back_mean, purple_back_mean, yellow_back_mean = \
                [np.mean(np.array(red_back)[:, 0]), np.mean(np.array(red_back)[:, 1])], [
                np.mean(np.array(green_back)[:, 0]), np.mean(np.array(green_back)[:, 1])], \
            [np.mean(np.array(blue_back)[:, 0]), np.mean(np.array(blue_back)[:, 1])], [
                np.mean(np.array(purple_back)[:, 0]), np.mean(np.array(purple_back)[:, 1])], \
            [np.mean(np.array(yellow_back)[:, 0]), np.mean(np.array(yellow_back)[:, 1])]
        elif len(red_back) != 0:
            red_back_mean, green_back_mean, blue_back_mean, purple_back_mean, yellow_back_mean = \
                [np.mean(np.array(red_back)[:, 0]), np.mean(np.array(red_back)[:, 1])], [
                    np.mean(np.array(green_back)[:, 0]), np.mean(np.array(green_back)[:, 1])], \
                [np.mean(np.array(blue_back)[:, 0]), np.mean(np.array(blue_back)[:, 1])], [0,0],[0,0]
        if len(purpleongreen) != 0:
            redongreen_mean, greenongreen_mean, blueongreen_mean, purpleongreen_mean, yellowongreen_mean = \
                [np.mean(np.array(redongreen)[:, 0]), np.mean(np.array(redongreen)[:, 1])], [
                np.mean(np.array(greenongreen)[:, 0]), np.mean(np.array(greenongreen)[:, 1])], \
            [np.mean(np.array(blueongreen)[:, 0]), np.mean(np.array(blueongreen)[:, 1])], [
                np.mean(np.array(purpleongreen)[:, 0]), np.mean(np.array(purpleongreen)[:, 1])], \
            [np.mean(np.array(yellowongreen)[:, 0]), np.mean(np.array(yellowongreen)[:, 1])]
        elif len(redongreen) != 0:
            redongreen_mean, greenongreen_mean, blueongreen_mean, purpleongreen_mean, yellowongreen_mean = \
                [np.mean(np.array(redongreen)[:, 0]), np.mean(np.array(redongreen)[:, 1])], [
                    np.mean(np.array(greenongreen)[:, 0]), np.mean(np.array(greenongreen)[:, 1])], \
                [np.mean(np.array(blueongreen)[:, 0]), np.mean(np.array(blueongreen)[:, 1])], [0,0], [0,0]
        if len(purpleonblue) != 0:
            redonblue_mean, greenonblue_mean, blueonblue_mean, purpleonblue_mean, yellowonblue_mean =  \
                [np.mean(np.array(redonblue)[:, 0]), np.mean(np.array(redonblue)[:, 1])], [
                np.mean(np.array(greenonblue)[:, 0]), np.mean(np.array(greenonblue)[:, 1])], \
            [np.mean(np.array(blueonblue)[:, 0]), np.mean(np.array(blueonblue)[:, 1])], [
                np.mean(np.array(purpleonblue)[:, 0]), np.mean(np.array(purpleonblue)[:, 1])], \
            [np.mean(np.array(yellowonblue)[:, 0]), np.mean(np.array(yellowonblue)[:, 1])]
        elif len(redonblue) != 0:
            redonblue_mean, greenonblue_mean, blueonblue_mean, purpleonblue_mean, yellowonblue_mean = \
                [np.mean(np.array(redonblue)[:, 0]), np.mean(np.array(redonblue)[:, 1])], [
                    np.mean(np.array(greenonblue)[:, 0]), np.mean(np.array(greenonblue)[:, 1])], \
                [np.mean(np.array(blueonblue)[:, 0]), np.mean(np.array(blueonblue)[:, 1])], [0,0], [0,0]
        if len(purpleonyellow) != 0:
            redonyellow_mean, greenonyellow_mean, blueonyellow_mean, purpleonyellow_mean, yellowonyellow_mean = \
                [np.mean(np.array(redonyellow)[:, 0]), np.mean(np.array(redonyellow)[:, 1])], [
                np.mean(np.array(greenonyellow)[:, 0]), np.mean(np.array(greenonyellow)[:, 1])], \
            [np.mean(np.array(blueonyellow)[:, 0]), np.mean(np.array(blueonyellow)[:, 1])], [
                np.mean(np.array(purpleonyellow)[:, 0]), np.mean(np.array(purpleonyellow)[:, 1])], \
            [np.mean(np.array(yellowonyellow)[:, 0]), np.mean(np.array(yellowonyellow)[:, 1])]
        elif len(redonyellow) != 0:
            redonyellow_mean, greenonyellow_mean, blueonyellow_mean, purpleonyellow_mean, yellowonyellow_mean = \
                [np.mean(np.array(redonyellow)[:, 0]), np.mean(np.array(redonyellow)[:, 1])], [
                    np.mean(np.array(greenonyellow)[:, 0]), np.mean(np.array(greenonyellow)[:, 1])], \
                [np.mean(np.array(blueonyellow)[:, 0]), np.mean(np.array(blueonyellow)[:, 1])], [0,0], [0,0]

        red = [grasp_red_mean, red_left_mean, red_right_mean, red_front_mean, red_back_mean, redongreen_mean, redonblue_mean, redonyellow_mean]
        green = [grasp_green_mean, green_left_mean, green_right_mean, green_front_mean, green_back_mean, greenongreen_mean,
               greenonblue_mean, greenonyellow_mean]
        blue = [grasp_blue_mean, blue_left_mean, blue_right_mean, blue_front_mean, blue_back_mean,
                 blueongreen_mean, blueonblue_mean, blueonyellow_mean]
        purple = [grasp_purple_mean, purple_left_mean, purple_right_mean, purple_front_mean, purple_back_mean,
                 purpleongreen_mean, purpleonblue_mean, purpleonyellow_mean]
        yellow = [grasp_yellow_mean, yellow_left_mean, yellow_right_mean, yellow_front_mean, yellow_back_mean,
                 yellowongreen_mean, yellowonblue_mean, yellowonyellow_mean]
        return red, green, blue, purple, yellow
    def plot_lang_latent(self, alph = 0.3, dir=None, lang_states=None, labels=None,pca_type="kpca", kernel='linear', colors=['red', 'green', 'blue', 'purple', 'yellow'], fn='lang_latent'):
        """
        colors = red, green, blue, purple, yellow
        labels and language states should be ordered appropriately
        """
        s, s_vecs = self.sents()

        labels_idx = []
        for i in range(len(labels)):
            for s in range(len(s_vecs)):
                if np.array_equal(labels[i], s_vecs[s]): #.numpy()
                    labels_idx.append(s)
        if pca_type == "kpca":
            kpca_2d = self.kpca(nc=2, kernel=kernel)
        elif pca_type == "pca":
            kpca_2d = self.pca(nc=2)
        lang_states = torch.from_numpy(lang_states)
        labels = torch.from_numpy(labels)
        lang_states_kpca2 = kpca_2d.fit_transform(lang_states[:].numpy())#
        if pca_type == 'kpca':
            var_values = kpca_2d.eigenvalues_ / sum(kpca_2d.eigenvalues_)
            print("variance explained ={}".format(var_values))
        kpca2d_1 = [lang_states_kpca2[i][0] for i in range(len(lang_states_kpca2))]
        kpca2d_2 = [lang_states_kpca2[i][1] for i in range(len(lang_states_kpca2))]
        normalized_kpca2d_1 = kpca2d_1 #self.normalize(pca2d_1)
        normalized_kpca2d_2 = kpca2d_2 # self.normalize(pca2d_2)
        lang_states_kpca2d = [[normalized_kpca2d_1[i], normalized_kpca2d_2[i]] for i in range(len(kpca2d_1))]

        clus = self.cluster_latent(n=40)
        # clustered_pca = clus.fit_transform(lang_states_pca2)
        # lang_states_pca2d=clustered_pca
        lang_states_red, lang_states_green, lang_states_blue, lang_states_yellow, lang_states_purple = [], [], [], [], []
        lang_states_red2, lang_states_green2, lang_states_blue2, lang_states_yellow2, lang_states_purple2 = [], [], [], [], []
        labels_red, labels_green, labels_blue, labels_purple, labels_yellow = [], [], [], [], []
        (lang_states_moveleft, lang_states_moveright, lang_states_movefront, lang_states_moveback,
         lang_states_grasp, lang_states_putongreen, lang_states_putonblue, lang_states_putonyellow) = [],[],[],[],[],[],[],[]
        (lang_states_moveleft2, lang_states_moveright2, lang_states_movefront2, lang_states_moveback2,
         lang_states_grasp2, lang_states_putongreen2, lang_states_putonblue2,
         lang_states_putonyellow2) = [], [], [], [], [], [], [], []
        (labels_left, labels_right, labels_front, labels_back,
         labels_grasp, labels_putongreen, labels_putonblue, labels_putonyellow) = [], [], [], [], [], [], [], []


        red_v = dp.lang_vec("red", max_len=1)
        green_v = dp.lang_vec("green", max_len=1)
        blue_v  = dp.lang_vec("blue", max_len=1)
        purple_v = dp.lang_vec("purple", max_len=1)
        yellow_v = dp.lang_vec("yellow", max_len=1)
        left_v = dp.lang_vec("left", max_len=1)
        right_v = dp.lang_vec("right", max_len=1)
        front_v = dp.lang_vec("front", max_len=1)
        back_v = dp.lang_vec("back", max_len=1)
        grasp_v = dp.lang_vec("grasp", max_len=1)
        put_v = dp.lang_vec("put", max_len=1)

        for i in range(len(lang_states)):
            if np.argmax(labels[i][0]) == np.argmax(grasp_v[0]):
                lang_states_grasp.append(lang_states_kpca2d[i][0])
                lang_states_grasp2.append(lang_states_kpca2d[i][1])
                labels_grasp.append(labels[i])
            elif np.argmax(labels[i][2]) == np.argmax(left_v[0]):
                lang_states_moveleft.append(lang_states_kpca2d[i][0])
                lang_states_moveleft2.append(lang_states_kpca2d[i][1])
                labels_left.append(labels[i])
            elif np.argmax(labels[i][2]) == np.argmax(right_v[0]):
                lang_states_moveright.append(lang_states_kpca2d[i][0])
                lang_states_moveright2.append(lang_states_kpca2d[i][1])
                labels_right.append(labels[i])
            elif np.argmax(labels[i][2]) == np.argmax(front_v[0]):
                lang_states_movefront.append(lang_states_kpca2d[i][0])
                lang_states_movefront2.append(lang_states_kpca2d[i][1])
                labels_front.append(labels[i])
            elif np.argmax(labels[i][2]) == np.argmax(back_v[0]):
                lang_states_moveback.append(lang_states_kpca2d[i][0])
                lang_states_moveback2.append(lang_states_kpca2d[i][1])
                labels_back.append(labels[i])
            elif np.argmax(labels[i][0]) == np.argmax(put_v[0]) and np.argmax(labels[i][3]) == np.argmax(green_v[0]):
                lang_states_putongreen.append(lang_states_kpca2d[i][0])
                lang_states_putongreen2.append(lang_states_kpca2d[i][1])
                labels_putongreen.append(labels[i])
            elif np.argmax(labels[i][0]) == np.argmax(put_v[0]) and np.argmax(labels[i][3]) == np.argmax(blue_v[0]):
                lang_states_putonblue.append(lang_states_kpca2d[i][0])
                lang_states_putonblue2.append(lang_states_kpca2d[i][1])
                labels_putonblue.append(labels[i])
            elif np.argmax(labels[i][0]) == np.argmax(put_v[0]) and np.argmax(labels[i][3]) == np.argmax(yellow_v[0]):
                lang_states_putonyellow.append(lang_states_kpca2d[i][0])
                lang_states_putonyellow2.append(lang_states_kpca2d[i][1])
                labels_putonyellow.append(labels[i])

        if len(lang_states_putonyellow) != 0  and len(lang_states_putonblue) != 0:  #5x8
            figa2d = plt.figure(figsize=(9,9))
            axgr = figa2d.add_subplot(331)
            axle = figa2d.add_subplot(332)
            axri = figa2d.add_subplot(333)
            axfr = figa2d.add_subplot(334)
            axba = figa2d.add_subplot(335)
            axpg = figa2d.add_subplot(336)
            axpb = figa2d.add_subplot(337)
            axpy = figa2d.add_subplot(338)

        elif len(lang_states_moveback) != 0 and len(lang_states_movefront) != 0:
            figa2d = plt.figure(figsize=(9,6))
            axgr = figa2d.add_subplot(231)
            axle = figa2d.add_subplot(232)
            axri = figa2d.add_subplot(233)
            axfr = figa2d.add_subplot(234)
            axba = figa2d.add_subplot(235)
            axpg = figa2d.add_subplot(236)
        else:
            figa2d = plt.figure(figsize=(9, 3))
            axgr = figa2d.add_subplot(131)
            axle = figa2d.add_subplot(132)
            axpg = figa2d.add_subplot(133)
        # print("len pg = {}".format(len(lang_states_putongreen)))
        col_gr, markers_gr = self.gen_markers(labels_grasp, colors)
        col_le, markers_le = self.gen_markers(labels_left, colors)
        col_pg, markers_pg = self.gen_markers(labels_putongreen, colors)
        if len(lang_states_moveback) != 0 and len(lang_states_movefront) != 0:
            col_ri, markers_ri = self.gen_markers(labels_right, colors)
            col_fr, markers_fr = self.gen_markers(labels_front, colors)
            col_ba, markers_ba = self.gen_markers(labels_back, colors)
            if len(lang_states_putonyellow) != 0  and len(lang_states_putonblue) != 0:
                col_pb, markers_pb = self.gen_markers(labels_putonblue,colors)
                col_py, markers_py = self.gen_markers(labels_putonyellow, colors)
        for i in range(len(lang_states_grasp)):
            axgr.scatter(lang_states_grasp[i], lang_states_grasp2[i], c=col_gr[i],
                         label=self.gen_lang(labels_grasp[i]), marker=markers_gr[i], alpha=alph)
            axgr.set_title('grasp')
            # axgr.set_xlim(0, 1)
            # axgr.set_ylim(0, 1)
            axle.scatter(lang_states_moveleft[i], lang_states_moveleft2[i], c=col_le[i],
                         label=self.gen_lang(labels_left[i]), marker=markers_le[i], alpha=alph)
            axle.set_title('move left')
            # axle.set_xlim(0, 1)
            # axle.set_ylim(0, 1)
            axpg.scatter(lang_states_putongreen[i], lang_states_putongreen2[i], c=col_pg[i],
                        label=self.gen_lang(labels_putongreen[i]), marker=markers_pg[i], alpha=alph)
            axpg.set_title('put on green')
            # axpg.set_xlim(0, 1)
            # axpg.set_ylim(0, 1)
            if len(lang_states_moveback) != 0 and len(lang_states_movefront) != 0:
                axri.scatter(lang_states_moveright[i], lang_states_moveright2[i], c=col_ri[i],
                            label=self.gen_lang(labels_right[i]), marker=markers_ri[i], alpha=alph)
                axri.set_title('move right')
                # axri.set_xlim(0, 1)
                # axri.set_ylim(0, 1)
                axfr.scatter(lang_states_movefront[i], lang_states_movefront2[i], c=col_fr[i],
                             label=self.gen_lang(labels_front[i]), marker=markers_fr[i], alpha=alph)
                axfr.set_title('move front')
                # axfr.set_xlim(0, 1)
                # axfr.set_ylim(0, 1)
                axba.scatter(lang_states_moveback[i], lang_states_moveback2[i], c=col_ba[i],
                             label=self.gen_lang(labels_back[i]), marker=markers_ba[i], alpha=alph)
                axba.set_title('move back')
                # axba.set_xlim(0, 1)
                # axba.set_ylim(0, 1)
                if len(lang_states_putonyellow) != 0 and len(lang_states_putonblue) != 0:
                    axpb.scatter(lang_states_putonblue[i], lang_states_putonblue2[i], c=col_pb[i],
                                 label=self.gen_lang(labels_putonblue[i]), marker=markers_pb[i], alpha=alph)
                    axpb.set_title('put on blue')
                    # axpb.set_xlim(0, 1)
                    # axpb.set_ylim(0, 1)
                    axpy.scatter(lang_states_putonyellow[i], lang_states_putonyellow2[i], c=col_py[i],
                                 label=self.gen_lang(labels_putonyellow[i]), marker=markers_py[i], alpha=alph)
                    axpy.set_title('put on yellow')
                    # axpy.set_xlim(0, 1)
                    # axpy.set_ylim(0, 1)

        mean_pca = [np.mean(lang_states_grasp), np.mean(lang_states_moveleft),
                    np.mean(lang_states_moveright), np.mean(lang_states_movefront),
                    np.mean(lang_states_moveback), np.mean(lang_states_putongreen),
                    np.mean(lang_states_putonblue), np.mean(lang_states_putonyellow),]
        std_pca = [np.std(lang_states_grasp), np.std(lang_states_moveleft),
                   np.std(lang_states_moveright), np.std(lang_states_movefront),
                   np.std(lang_states_moveback), np.std(lang_states_putongreen),
                   np.std(lang_states_putonblue),np.std(lang_states_putonyellow)]
        mean_pca2 = [np.mean(lang_states_grasp2), np.mean(lang_states_moveleft2),
                    np.mean(lang_states_moveright2), np.mean(lang_states_movefront2),
                    np.mean(lang_states_moveback2), np.mean(lang_states_putongreen2),
                    np.mean(lang_states_putonblue2), np.mean(lang_states_putonyellow2),]
        std_pca2 = [np.std(lang_states_grasp2), np.std(lang_states_moveleft2),
                   np.std(lang_states_moveright2), np.std(lang_states_movefront2),
                   np.std(lang_states_moveback2), np.std(lang_states_putongreen2),
                   np.std(lang_states_putonblue2),np.std(lang_states_putonyellow2)]
        # print(len(lang_states_grasp), len(lang_states_grasp2))
        print("mean={}, std={}".format(mean_pca, std_pca))
        print("mean2={}, std2={}".format(mean_pca2, std_pca2))

        figa2d.tight_layout()
        # fig_l.subplots_adjust(right=0.05)
        # self.legend_without_duplicate_labels(axy, loc='lower center', fontsize=5, n_col=2, bbox_to_anchor=(1.5, 0.5))
        # plt.show()
        plt.savefig("{}/{}.png".format(dir, fn+"pca2d_verbs_kclus"), bbox_inches='tight')
        plt.close()


        for i in range(len(lang_states)):
            if np.argmax(labels[i][1]) == np.argmax(red_v[0]):
                lang_states_red.append(lang_states_kpca2d[i][0])
                lang_states_red2.append(lang_states_kpca2d[i][1])
                labels_red.append(labels[i])
            elif np.argmax(labels[i][1]) == np.argmax(green_v[0]):
                lang_states_green.append(lang_states_kpca2d[i][0])
                lang_states_green2.append(lang_states_kpca2d[i][1])
                labels_green.append(labels[i])
            elif np.argmax(labels[i][1]) == np.argmax(blue_v[0]):
                lang_states_blue.append(lang_states_kpca2d[i][0])
                lang_states_blue2.append(lang_states_kpca2d[i][1])
                labels_blue.append(labels[i])
            elif np.argmax(labels[i][1]) == np.argmax(purple_v[0]):
                lang_states_purple.append(lang_states_kpca2d[i][0])
                lang_states_purple2.append(lang_states_kpca2d[i][1])
                labels_purple.append(labels[i])
            elif np.argmax(labels[i][1]) == np.argmax(yellow_v[0]):
                lang_states_yellow.append(lang_states_kpca2d[i][0])
                lang_states_yellow2.append(lang_states_kpca2d[i][1])
                labels_yellow.append(labels[i])
        red_pca = lang_states_red
        red_pca2 = lang_states_red2
        green_pca = lang_states_green
        green_pca2 = lang_states_green2
        blue_pca = lang_states_blue
        blue_pca2 = lang_states_blue2
        purple_pca = lang_states_purple
        purple_pca2 = lang_states_purple2
        yellow_pca = lang_states_yellow
        yellow_pca2 = lang_states_yellow2

        # mean values of pca
        red_mean, green_mean, blue_mean, purple_mean, yellow_mean = self.get_mean_latent(kpca2d_1, kpca2d_2, labels)

        red_pca_m = np.array(red_mean)[:, 0]
        red_pca2_m = np.array(red_mean)[:, 1]
        green_pca_m = np.array(green_mean)[:, 0]
        green_pca2_m = np.array(green_mean)[:, 1]
        blue_pca_m = np.array(blue_mean)[:, 0]
        blue_pca2_m = np.array(blue_mean)[:, 1]
        purple_pca_m = np.array(purple_mean)[:, 0]
        purple_pca2_m = np.array(purple_mean)[:, 1]
        yellow_pca_m = np.array(yellow_mean)[:, 0]
        yellow_pca2_m = np.array(yellow_mean)[:, 1]
        labels_red_m = [dp.lang_vec("grasp red .", max_len=5)[0], dp.lang_vec("move red left .", max_len=5)[0],
                        dp.lang_vec("move red right .", max_len=5)[0], dp.lang_vec("move red front .", max_len=5)[0],
                        dp.lang_vec("move red back .", max_len=5)[0], dp.lang_vec("put red on green .", max_len=5)[0],
                        dp.lang_vec("put red on blue .", max_len=5)[0],
                        dp.lang_vec("put red on yellow .", max_len=5)[0]]
        labels_green_m = [dp.lang_vec("grasp green .", max_len=5)[0], dp.lang_vec("move green left .", max_len=5)[0],
                          dp.lang_vec("move green right .", max_len=5)[0],
                          dp.lang_vec("move green front .", max_len=5)[0],
                          dp.lang_vec("move green back .", max_len=5)[0],
                          dp.lang_vec("put green on green .", max_len=5)[0],
                          dp.lang_vec("put green on blue .", max_len=5)[0],
                          dp.lang_vec("put green on yellow .", max_len=5)[0]]
        labels_blue_m = [dp.lang_vec("grasp blue .", max_len=5)[0], dp.lang_vec("move blue left .", max_len=5)[0],
                         dp.lang_vec("move blue right .", max_len=5)[0], dp.lang_vec("move blue front .", max_len=5)[0],
                         dp.lang_vec("move blue back .", max_len=5)[0],
                         dp.lang_vec("put blue on green .", max_len=5)[0],
                         dp.lang_vec("put blue on blue .", max_len=5)[0],
                         dp.lang_vec("put blue on yellow .", max_len=5)[0]]
        labels_purple_m = [dp.lang_vec("grasp purple .", max_len=5)[0], dp.lang_vec("move purple left .", max_len=5)[0],
                           dp.lang_vec("move purple right .", max_len=5)[0],
                           dp.lang_vec("move purple front .", max_len=5)[0],
                           dp.lang_vec("move purple back .", max_len=5)[0],
                           dp.lang_vec("put purple on green .", max_len=5)[0],
                           dp.lang_vec("put purple on blue .", max_len=5)[0],
                           dp.lang_vec("put purple on yellow .", max_len=5)[0]]
        labels_yellow_m = [dp.lang_vec("grasp yellow .", max_len=5)[0], dp.lang_vec("move yellow left .", max_len=5)[0],
                           dp.lang_vec("move yellow right .", max_len=5)[0],
                           dp.lang_vec("move yellow front .", max_len=5)[0],
                           dp.lang_vec("move yellow back .", max_len=5)[0],
                           dp.lang_vec("put yellow on green .", max_len=5)[0],
                           dp.lang_vec("put yellow on blue .", max_len=5)[0],
                           dp.lang_vec("put yellow on yellow .", max_len=5)[0]]
        #
        # if len(lang_states_moveback) != 0 and len(lang_states_movefront) != 0:
        #     if len(lang_states_putonyellow) != 0 and len(lang_states_putonblue) != 0:
        #         labels_red_m = [dp.lang_vec("grasp red .", max_len=5)[0], dp.lang_vec("move red left .", max_len=5)[0],
        #                         dp.lang_vec("move red right .", max_len=5)[0], dp.lang_vec("move red front .", max_len=5)[0],
        #                         dp.lang_vec("move red back .", max_len=5)[0],dp.lang_vec("put red on green .", max_len=5)[0],
        #                         dp.lang_vec("put red on blue .", max_len=5)[0], dp.lang_vec("put red on yellow .", max_len=5)[0]]
        #         labels_green_m = [dp.lang_vec("grasp green .", max_len=5)[0], dp.lang_vec("move green left .", max_len=5)[0],
        #                         dp.lang_vec("move green right .", max_len=5)[0], dp.lang_vec("move green front .", max_len=5)[0],
        #                         dp.lang_vec("move green back .", max_len=5)[0],dp.lang_vec("put green on green .", max_len=5)[0],
        #                         dp.lang_vec("put green on blue .", max_len=5)[0], dp.lang_vec("put green on yellow .", max_len=5)[0]]
        #         labels_blue_m = [dp.lang_vec("grasp blue .", max_len=5)[0], dp.lang_vec("move blue left .", max_len=5)[0],
        #                         dp.lang_vec("move blue right .", max_len=5)[0], dp.lang_vec("move blue front .", max_len=5)[0],
        #                         dp.lang_vec("move blue back .", max_len=5)[0],dp.lang_vec("put blue on green .", max_len=5)[0],
        #                         dp.lang_vec("put blue on blue .", max_len=5)[0], dp.lang_vec("put blue on yellow .", max_len=5)[0]]
        #         labels_purple_m = [dp.lang_vec("grasp purple .", max_len=5)[0], dp.lang_vec("move purple left .", max_len=5)[0],
        #                         dp.lang_vec("move purple right .", max_len=5)[0], dp.lang_vec("move purple front .", max_len=5)[0],
        #                         dp.lang_vec("move purple back .", max_len=5)[0],dp.lang_vec("put purple on green .", max_len=5)[0],
        #                         dp.lang_vec("put purple on blue .", max_len=5)[0], dp.lang_vec("put purple on yellow .", max_len=5)[0]]
        #         labels_yellow_m = [dp.lang_vec("grasp yellow .", max_len=5)[0], dp.lang_vec("move yellow left .", max_len=5)[0],
        #                         dp.lang_vec("move yellow right .", max_len=5)[0], dp.lang_vec("move yellow front .", max_len=5)[0],
        #                         dp.lang_vec("move yellow back .", max_len=5)[0],dp.lang_vec("put yellow on green .", max_len=5)[0],
        #                         dp.lang_vec("put yellow on blue .", max_len=5)[0], dp.lang_vec("put yellow on yellow .", max_len=5)[0]]
        #
        #     else:
        #         labels_red_m = [dp.lang_vec("grasp red .", max_len=5)[0], dp.lang_vec("move red left .", max_len=5)[0],
        #                         dp.lang_vec("move red right .", max_len=5)[0], dp.lang_vec("move red front .", max_len=5)[0],
        #                         dp.lang_vec("move red back .", max_len=5)[0],dp.lang_vec("put red on green .", max_len=5)[0]]
        #         labels_green_m = [dp.lang_vec("grasp green .", max_len=5)[0], dp.lang_vec("move green left .", max_len=5)[0],
        #                         dp.lang_vec("move green right .", max_len=5)[0], dp.lang_vec("move green front .", max_len=5)[0],
        #                         dp.lang_vec("move green back .", max_len=5)[0],dp.lang_vec("put green on green .", max_len=5)[0]]
        #         labels_blue_m = [dp.lang_vec("grasp blue .", max_len=5)[0], dp.lang_vec("move blue left .", max_len=5)[0],
        #                         dp.lang_vec("move blue right .", max_len=5)[0], dp.lang_vec("move blue front .", max_len=5)[0],
        #                         dp.lang_vec("move blue back .", max_len=5)[0],dp.lang_vec("put blue on green .", max_len=5)[0]]
        #         labels_purple_m = [dp.lang_vec("grasp purple .", max_len=5)[0], dp.lang_vec("move purple left .", max_len=5)[0],
        #                         dp.lang_vec("move purple right .", max_len=5)[0], dp.lang_vec("move purple front .", max_len=5)[0],
        #                         dp.lang_vec("move purple back .", max_len=5)[0],dp.lang_vec("put purple on green .", max_len=5)[0]]
        #         labels_yellow_m = [dp.lang_vec("grasp yellow .", max_len=5)[0], dp.lang_vec("move yellow left .", max_len=5)[0],
        #                         dp.lang_vec("move yellow right .", max_len=5)[0], dp.lang_vec("move yellow front .", max_len=5)[0],
        #                         dp.lang_vec("move yellow back .", max_len=5)[0],dp.lang_vec("put yellow on green .", max_len=5)[0]]
        # else:
        #     labels_red_m = [dp.lang_vec("grasp red .", max_len=5)[0], dp.lang_vec("move red left .", max_len=5)[0],
        #                     dp.lang_vec("put red on green .", max_len=5)[0]]
        #     labels_green_m = [dp.lang_vec("grasp green .", max_len=5)[0],
        #                       dp.lang_vec("move green left .", max_len=5)[0],
        #                       dp.lang_vec("put green on green .", max_len=5)[0]]
        #     labels_blue_m = [dp.lang_vec("grasp blue .", max_len=5)[0], dp.lang_vec("move blue left .", max_len=5)[0],
        #                      dp.lang_vec("put blue on green .", max_len=5)[0]]
        #     labels_purple_m = [dp.lang_vec("grasp purple .", max_len=5)[0],
        #                        dp.lang_vec("move purple left .", max_len=5)[0],
        #                        dp.lang_vec("put purple on green .", max_len=5)[0]]
        #     labels_yellow_m = [dp.lang_vec("grasp yellow .", max_len=5)[0],
        #                        dp.lang_vec("move yellow left .", max_len=5)[0],
        #                        dp.lang_vec("put yellow on green .", max_len=5)[0]]

        col_rm, markers_rm = self.gen_markers(labels_red_m, ['red'])
        col_gm, markers_gm = self.gen_markers(labels_green_m, ['green'])
        col_bm, markers_bm = self.gen_markers(labels_blue_m, ['blue'])
        if len(purple_pca_m) != 0  and len(yellow_pca_m) != 0:
            col_pm, markers_pm = self.gen_markers(labels_purple_m, ['purple'])
            col_ym, markers_ym = self.gen_markers(labels_yellow_m, ['yellow'])
        print("no. of markers={}".format(len(markers_rm)))

        if len(purple_pca) != 0  and len(yellow_pca) != 0:
            fig2d = plt.figure(figsize=(15, 3))
            axr = fig2d.add_subplot(151)
            axg = fig2d.add_subplot(152)
            axb = fig2d.add_subplot(153)
            axp = fig2d.add_subplot(154)
            axy = fig2d.add_subplot(155)
            # to plot mean values
            fig2dm = plt.figure(figsize=(15, 3))
            axrm = fig2dm.add_subplot(151)
            axgm = fig2dm.add_subplot(152)
            axbm = fig2dm.add_subplot(153)
            axpm = fig2dm.add_subplot(154)
            axym = fig2dm.add_subplot(155)

        else:
            fig2d = plt.figure(figsize=(9, 3))
            axr = fig2d.add_subplot(131)
            axg = fig2d.add_subplot(132)
            axb = fig2d.add_subplot(133)

            fig2dm = plt.figure(figsize=(9, 3))
            axrm = fig2dm.add_subplot(131)
            axgm = fig2dm.add_subplot(132)
            axbm = fig2dm.add_subplot(133)
        # print(len(red_pca_m))
        # exit()
        mean_marker_size=80
        for i in range(len(red_pca_m)):
            if red_pca_m[i] == 0 and i != len(red_pca_m)-1:
                red_pca_m[i] = red_pca_m[i+1]
                red_pca2_m[i] = red_pca2_m[i+1]

            if green_pca_m[i] == 0 and i != len(red_pca_m)-1:
                green_pca_m[i] = green_pca_m[i+1]
                green_pca2_m[i] = green_pca2_m[i+1]

            if blue_pca_m[i] == 0 and i != len(red_pca_m)-1:
                blue_pca_m[i] = blue_pca_m[i+1]
                blue_pca2_m[i] = blue_pca2_m[i+1]
            elif red_pca_m[i] !=0:
                axrm.scatter(red_pca_m[i], red_pca2_m[i], c='r', marker=markers_rm[i], s=mean_marker_size, alpha=0.8)
                axgm.scatter(green_pca_m[i], green_pca2_m[i], c='g', marker=markers_gm[i], s=mean_marker_size, alpha=0.8)
                axbm.scatter(blue_pca_m[i], blue_pca2_m[i], c='b', marker=markers_bm[i], s=mean_marker_size, alpha=0.8)
            if len(purple_pca) != 0 and len(yellow_pca) != 0:
                if purple_pca_m[i] == 0 and i != len(red_pca_m)-1:
                    purple_pca_m[i] = purple_pca_m[i + 1]
                    purple_pca2_m[i] = purple_pca2_m[i + 1]
                if yellow_pca_m[i] == 0 and i != len(red_pca_m)-1:
                    yellow_pca_m[i] = yellow_pca_m[i + 1]
                    yellow_pca2_m[i] = yellow_pca2_m[i + 1]
                elif purple_pca_m[i] != 0:
                    axpm.scatter(purple_pca_m[i], purple_pca2_m[i], c='m', marker=markers_pm[i], s=mean_marker_size, alpha=0.8)
                    axym.scatter(yellow_pca_m[i], yellow_pca2_m[i], c='y', marker=markers_ym[i], s=mean_marker_size, alpha=0.8)


        fig2dm.tight_layout()
        fig2dm.savefig("{}/{}.png".format(dir, fn + "pca2d_nouns_mean"), bbox_inches='tight')


        col_r, markers_r = self.gen_markers(labels_red, ['red'])
        col_g, markers_g = self.gen_markers(labels_green, ['green'])
        col_b, markers_b = self.gen_markers(labels_blue, ['blue'])
        if len(purple_pca) != 0  and len(yellow_pca) != 0:
            col_p, markers_p = self.gen_markers(labels_purple, ['purple'])
            col_y, markers_y = self.gen_markers(labels_yellow, ['yellow'])
        for i in range(len(red_pca)):
            axr.scatter(red_pca[i], red_pca2[i], c='r', label=self.gen_lang(labels_red[i]), marker=markers_r[i], alpha=alph)

            # axr.set_title('red')
            # axr.set_xlim(0, 1)
            # axr.set_ylim(0, 1)
            axg.scatter(green_pca[i], green_pca2[i], c='g', label=self.gen_lang(labels_green[i]), marker=markers_g[i], alpha=alph)

            # axg.set_title("green")
            # axg.set_xlim(0, 1)
            # axg.set_ylim(0, 1)
            axb.scatter(blue_pca[i], blue_pca2[i], c='b', label=self.gen_lang(labels_blue[i]), marker=markers_b[i], alpha=alph)

            # axb.set_title("blue")
            # axb.set_xlim(0, 1)
            # axb.set_ylim(0, 1)
            if len(purple_pca) != 0 and len(yellow_pca) != 0:
                axp.scatter(purple_pca[i], purple_pca2[i], c='m', label=self.gen_lang(labels_purple[i]), marker=markers_p[i], alpha=alph)

                # axp.set_title("purple")
                # axp.set_xlim(0, 1)
                # axp.set_ylim(0, 1)
                axy.scatter(yellow_pca[i], yellow_pca2[i], c='y', label=self.gen_lang(labels_yellow[i]), marker=markers_y[i], alpha=alph)
                # axy.set_title("yellow")
                # axy.set_xlim(0, 1)
                # axy.set_ylim(0, 1)

        fig2d.tight_layout()

        fig2d.savefig("{}/{}.png".format(dir, fn+"pca2d_nouns"), bbox_inches='tight')


        plt.close()

        #
        n = 3
        if lang_states.shape[-1] < n:
            nc = lang_states.shape[-1]
            iso_l = self.isomap(nc=nc)
            iso_lang = iso_l.fit_transform(lang_states[:].numpy())
            kpca_l = self.kpca(nc=nc, kernel='linear')
            kpca_lang = kpca_l.fit_transform(lang_states[:].numpy())
            pca_l = self.pca(nc)
            pca_lang = pca_l.fit_transform(lang_states[:].numpy())
            umap_reducer = self.umap(nc=3, n_neighbors=80)
            umap_lang = umap_reducer.fit_transform(lang_states[:].numpy())
            tsne_l = self.tsne(nc=3)
            tsne_lang = tsne_l.fit_transform(lang_states.numpy())
        else:
            nc = n
            iso_l = self.isomap(nc=nc)
            iso_lang = iso_l.fit_transform(lang_states[:].numpy())
            kpca_l = self.kpca(nc=nc, kernel='cosine')
            kpca_lang = kpca_l.fit_transform(lang_states[:].numpy())
            pca_lang = lang_states.numpy()
            umap_reducer = self.umap(nc=3)
            umap_lang = umap_reducer.fit_transform(lang_states[:].numpy())
            tsne_l = self.tsne(nc=3)
            tsne_lang = tsne_l.fit_transform(lang_states.numpy())


        # clustered_sne = clus.fit_transform(tsne_lang)
        clustered_umap = clus.fit_transform(umap_lang)
        #
        # print("nc={}".format(nc))
        # print("shape = {}".format(lang_states.numpy().shape))
        # print("umapshape = {}".format(umap_lang.shape))
        # print("labels shape = {}".format(labels.shape))
        #
        col, markers = self.gen_markers(labels, colors)
        # fig_l = plt.figure()
        # ax = fig_l.add_subplot(111, projection='3d')
        # # ax_p1 = fig_l.add_subplot(211)
        # # ax_p2 = fig_l.add_subplot(212)
        #
        label_str = []
        for l in range(len(labels)):
            label_str.append(self.gen_lang(labels[l]))
        # for i in range(len(pca_lang)):
        #     # ax_p1.scatter(pca_lang[i][0], pca_lang[i][1], c=col[i], label=label_str[i], marker=markers[i])
        #     # ax_p2.scatter(pca_lang[i][2], pca_lang[i][3], c=col[i], label=label_str[i], marker=markers[i])
        #     #
        #     # box = ax_p2.get_position()
        #     # ax_p2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #     ax.scatter(pca_lang[i][0], pca_lang[i][1], pca_lang[i][2], c=col[i], label=label_str[i], marker=markers[i])
        #     # ax.scatter(clustered_pca[i][0], clustered_pca[i][1], clustered_pca[i][2], c=col[i], label=label_str[i], marker=markers[i])
        #
        #     # ax.plot(pca_lang[i][0], pca_lang[i][1], pca_lang[i][2], 'o', ms=20, mec='red', mfc='none', mew=2)
        # # plt.tight_layout()
        # fig_l.tight_layout()
        # # fig_l.subplots_adjust(right=0.05)
        # self.legend_without_duplicate_labels(ax, loc='upper center', fontsize=5, n_col=2, bbox_to_anchor=(1.5, 0.5))
        # # plt.show()
        # plt.savefig("{}/{}.png".format(dir, fn), bbox_inches='tight')
        # plt.close()
        #
        # fig_k = plt.figure()
        # axk = fig_k.add_subplot(111, projection='3d')
        # for i in range(len(pca_lang)):
        #     axk.scatter(kpca_lang[i][0], kpca_lang[i][1], kpca_lang[i][2], c=col[i], label=label_str[i], marker=markers[i])
        # fig_k.tight_layout()
        # self.legend_without_duplicate_labels(axk, loc='upper center', fontsize=5, n_col=2, bbox_to_anchor=(1.3, 0.5))
        # plt.savefig("{}/{}.png".format(dir, fn+"kpca"), bbox_inches='tight')
        # plt.close()
        #
        # fig_i = plt.figure()
        # axi = fig_i.add_subplot(111, projection='3d')
        # for i in range(len(pca_lang)):
        #     axi.scatter(iso_lang[i][0], iso_lang[i][1], iso_lang[i][2], c=col[i], label=label_str[i], marker=markers[i])
        # fig_i.tight_layout()
        # self.legend_without_duplicate_labels(axi, loc='upper center', fontsize=5, n_col=2, bbox_to_anchor=(1.3, 0.5))
        # plt.savefig("{}/{}.png".format(dir, fn+"iso"), bbox_inches='tight')
        # plt.close()
        #
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111, projection='3d')
        for i in range(len(umap_lang)):
            # ax1.scatter(umap_lang[i][0], umap_lang[i][1], umap_lang[i][2], c=col[i], label=label_str[i], marker=markers[i])
            ax1.scatter(clustered_umap[i][0], clustered_umap[i][1], clustered_umap[i][2], c=col[i],
                        label=label_str[i], marker=markers[i])
        fig2.tight_layout()
        self.legend_without_duplicate_labels(ax1, loc='upper center', fontsize=5, n_col=2)
        plt.savefig("{}/{}.png".format(dir, fn+"umap"), bbox_inches='tight')
        plt.close()
        #
        # fig3 = plt.figure()
        # ax2 = fig3.add_subplot(111, projection='3d')
        # for i in range(len(tsne_lang)):
        #     # ax2.scatter(tsne_lang[i][0], tsne_lang[i][1], tsne_lang[i][2], c=col[i], label=label_str[i], marker=markers[i])
        #     ax2.scatter(clustered_sne[i][0], clustered_sne[i][1], clustered_sne[i][2], c=col[i], label=label_str[i], marker=markers[i])
        # fig3.tight_layout()
        # self.legend_without_duplicate_labels(ax2, loc='upper center', fontsize=5, n_col=2)
        # plt.savefig("{}/{}.png".format(dir, fn+"tsne"), bbox_inches='tight')
        # plt.close()


        save_dict = {'pca_lang_latent_states': lang_states_kpca2d, 'lang_labels': labels,
                     "label_index_list": labels_idx, 'lang_latent_states': lang_states.numpy()}
        np.savez(dir+"/latentplotdata", **save_dict)

########
    def usefn(self):
        dir = "exp_comp1_5x8_10_101/v-1/trainstate4500/rep0_aea5ec92-bb9d-41ef-9c9f-7d4c2b91aef5"
        data = np.load(dir + "/latentplotdata.npz")
        lang_states = data['lang_latent_states']
        labels = data['lang_labels']

        from utils import IO
        io = IO(dir)
        io.plot_lang_latent(dir=dir,
                            lang_states=lang_states, labels=labels, colors=['red', 'green', 'blue', 'purple', 'yellow'],
                            fn='/lang_latent2'
                            )
########
    def plot_latentstates(self, states, nc=5, name='pvrnn_d.png'):
        """
        return pca plot of latent state of activity of layers in the network
        todo:it should be able to visualize prior and posterior activity of PVRNN as well
        """
        # ToDo
        fig = plt.figure(figsize=(10,5))
        r, c = len(states[0]), 1
        new_states, states_pca = self.pca_states(states, nc=nc)
        print(states_pca[0].shape)
        for l in range(len(states[0])):
            ax = fig.add_subplot(r,c,l+1)
            ax.plot(states_pca[l])
        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(self.work_dir, name))
        plt.close()

    def pca_init(self, pca, train_lang, test_lang):
        lang = torch.cat((train_lang, test_lang))
        pca_lang = pca.fit_transform(lang.numpy().reshape(len(lang), -1))
        train_lang_pca, test_lang_pca = pca_lang[:len(train_lang), :], pca_lang[len(train_lang):, :]
        return torch.from_numpy(train_lang_pca), torch.from_numpy(test_lang_pca)

    def load_model(self, model, **model_args):
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        return model

    def load_weights(self, model, weights_path, ignore_weights=None):
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.print_log('Load weights from {}.'.format(weights_path))
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in weights.items()])
        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                self.print_log('Filter [{}] remove weights [{}].'.format(i,n))

        for w in weights:
            self.print_log('Load weights [{}].'.format(w))

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                self.print_log('Can not find weights [{}].'.format(d))
            state.update(weights)
            model.load_state_dict(state)
        return model

    def save_pkl(self, result, filename):
        with open('{}/{}'.format(self.work_dir, filename), 'wb') as f:
            pickle.dump(result, f)

    def save_h5(self, result, filename):
        with h5py.File('{}/{}'.format(self.work_dir, filename), 'w') as f:
            for k in result.keys():
                f[k] = result[k]

    def save_model(self, model, name):
        model_path = '{}/{}'.format(self.work_dir, name)
        state_dict = model.state_dict()
        weights = OrderedDict([[''.join(k.split('module.')),
                                v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        self.print_log('The model has been saved as {}.'.format(model_path))

    def load_checkpoint(self, name, dev1="cuda:0"):
        checkpoint_path = '{}/{}'.format(self.work_dir, name)
        if torch.cuda.is_available():
            state = torch.load(checkpoint_path, map_location=torch.device(dev1))
        else:
            state = torch.load(checkpoint_path, map_location = 'cpu')
        self.print_log('The checkpoint has been loaded from {}.'.format(checkpoint_path))

        return state

    def load_weights_from_checkpoint(self, model, state):
        model.load_state_dict(state['model'])
        return model
    def load_from_checkpoint(self, state, param='prior'):
        return state[param]

    def load_optimizer(self, state, param='prior'):
        return state['optimizer'][param]

    def save_checkpoint(self, state, name):
        checkpoint_path = '{}/{}'.format(self.work_dir, name)
        torch.save(state, checkpoint_path)
        self.print_log('The checkpoint has been saved as {}.'.format(checkpoint_path))

    # def save_gif(self, images, name):
    #     gif = images[0]
    #     gif_path = '{}/{}.gif'.format(self.work_dir, name)

    #     gif.save(fp=gif_path, format='gif', save_all=True, append_images=images[1:], duration=80)

    def save_video(self, images, name, figsize=(32,32)):

        Writer = animation.writers['ffmpeg']

        #writer_rgb = Writer(fps=25, metadata=dict(artist='rgb video', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18']), codec='libx264')  # , bitrate=1800 , codec='ffv1'
        writer_rgb = Writer(fps=25, metadata=dict(artist='rgb video',extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '0']), bitrate=1800,
                            codec='libx264')  # , bitrate=1800 , codec='ffv1'
        print("figsize", figsize)
        movFig_rgb = plt.figure(1, figsize=figsize)
        movFig_rgb.set_dpi(300)
        # plt.show()
        movFig_rgb_ax = movFig_rgb.add_axes([0, 0, 1, 1])
        video_path = '{}/{}.mp4'.format(self.work_dir, name)
        writer_rgb.setup(movFig_rgb, video_path, dpi=300)

        for frames in range(images.shape[0] - 1):
            npimg=np.squeeze(np.moveaxis(images[frames+1].detach().cpu().numpy(),0,-1))
            # Convert the RGB image data to image and visualize it:
            #image_byte_array = array.array('b',npimg)
            #im = Image.frombuffer('RGB', (npimg.shape[0],npimg.shape[1]), image_byte_array.tobytes(), 'raw', 'RGB', 0, 1)
            #im = Image.fromarray(np.squeeze(np.moveaxis(  images[frames+1].cpu().numpy(),0,-1)), 'RGB')
            im = Image.fromarray(np.squeeze(np.moveaxis(np.array(images[frames + 1].detach().cpu().numpy() * 255, dtype='uint8'), 0, -1)), 'RGB')
            movFig_rgb_ax.clear()
            movFig_rgb_ax.imshow(im, origin='upper')
            movFig_rgb_ax.axis('off')
            writer_rgb.grab_frame()

        writer_rgb.finish()
        plt.close()

    def save_gif(self, images, name):
        import torchvision
        gif_path = '{}/{}.png'.format(self.work_dir, name)
        torchvision.utils.save_image(images, gif_path, nrow=11)

    def legend_without_duplicate_labels(self, ax, loc=0, fontsize=10, n_col=5, bbox_to_anchor=(1.02, 1)):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=n_col)

    def save_motor(self, m_pred, m_target, name, softmax_config, title=""):
        """
        plot motor trajectory
        """
        m_t = m_target #np.load(name + "/m_target.npy", allow_pickle=True)
        m = m_pred #np.load(name + "/m_pred.npy", allow_pickle=True)

        m_ = dp.convert_from_softmax(softmax_config[0], np.exp(m))
        m_tar = dp.convert_from_softmax(softmax_config[0], m_t)
        # m_ = dp.renormalize_mot(m_)
        # m_tar = dp.renormalize_mot(m_tar)
        fig_m, axs = plt.subplots(2,1,figsize=(16,4), gridspec_kw={'height_ratios': [5, 1.2]})
        ax1 = axs[0] #= fig_m.add_subplot(211)
        ax2 = axs[1] # fig_m.add_subplot(212)
        label_ = []
        label_t = []
        label_d = []
        diff = []
        for i in range(len(m_)):
            label_d.append(i)
            diff.append(abs(m_[i] - m_tar[i]).mean())
        col = ['r', 'g', 'b', 'y', 'm', 'brown']
        for j in range(len(m_[0])):
            label_.append("pred. joint {}".format(j))
            label_t.append("target joint {}".format(j))
            line1 = ax1.plot(m_[:, j], color=col[j],label=label_[j])
            line2 = ax1.plot(m_tar[:, j], '--', color=col[j], label=label_t[j])

        # ax1.set_ylabel('joint angle')
        ax1.set_xlim(0, 49)
        ax1.set_xticks([])
        # ax1.set_yticks(np.arange(0.1, 0.9))

        self.legend_without_duplicate_labels(ax1, loc="upper left", n_col=1, bbox_to_anchor=(1, 1))

        ax2.bar(label_d, diff, color='red')  # ax2.bar(diff, diff, color='red')
        ax2.set_ylim(0,0.2)
        ax2.set_xlim(0, 49)
        # ax2.set_xlabel('time steps')
        ax2.set_xticks(np.arange(0, len(label_d), 10))
        # ax2.set_ylim(top=0.2)
        # plt.title(title)
        plt.subplots_adjust(hspace=0.1)
        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(self.work_dir, name))
        plt.close()
        return np.mean(np.array(diff))

    def mot(self, m, softmax_config, name):

        m_ = dp.convert_from_softmax(softmax_config[0], np.exp(m))

        # m_ = dp.renormalize_mot(m_)
        fig_m, axs = plt.subplots(1,1,figsize=(10,5))
        col = ['r', 'g', 'b', 'y', 'm', 'brown']
        for j in range(len(m_[0])):
            line1 = axs.plot(m_[:, j], color=col[j])
        axs.set_xticks([])
        axs.set_yticks([])
        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(self.work_dir, name))

    def vis_mot_fig(self, motor, vision, softmax_config, name='fig.png', v_seq_len=12):
        """
        save final figure with vision predicion, target, difference, memory content
        motor prediction, target and differcence.
        motor: dictionary with motor results
        vision: dictionary with vision results
        v_seq_len: len of vision prediction images to be plotted since all of them cannot fit in the figure.
        """
        m_pred = motor['m_pred']
        m_target = motor['m_target']
        v_pred = vision['v_pred']
        v_target = vision['v_target']
        l0mem = vision['l0mem']
        l1mem_masked = vision['l1mem_masked']
        m_t = m_target  # np.load(name + "/m_target.npy", allow_pickle=True)
        m = m_pred  # np.load(name + "/m_pred.npy", allow_pickle=True)
        m_ = dp.convert_from_softmax(softmax_config[0], np.exp(m))
        m_tar = dp.convert_from_softmax(softmax_config[0], m_t)
        label = []
        diff = []
        for i in range(len(m_)):
            label.append(i)
            diff.append(abs(m_[i] - m_tar[i]).mean())
        fig_a = plt.figure(2, figsize=(35, 10))
        axm1 = fig_a.add_subplot(716)
        axm2 = fig_a.add_subplot(717)
        line1 = axm1.plot(m_, c='r', label='prediction')
        line2 = axm1.plot(m_tar, '--', c='b', label='ground truth')
        self.legend_without_duplicate_labels(axm1)
        axm2.bar(label, diff, color='red')  # ax2.bar(diff, diff, color='red')
        axm2.set_ylim(0, 10)

        axv1 = fig_a.add_subplot(711)
        axv2 = fig_a.add_subplot(712)
        axv3 = fig_a.add_subplot(713)
        axv4 = fig_a.add_subplot(714)
        axv5 = fig_a.add_subplot(715)

        # ToDo


    def gen_lang(self, lang_vec):
        """
        take language vector and give language string
        vector from network will not be precise, so we argmax to generate words from output.
        """
        l_ = []
        for i in range(20):
            a = torch.zeros(20)
            a[i] = 1
            l_.append(a)
        l_dict = {}
        l_dict['.'] = l_[0]
        l_dict['touch'] = l_[1]
        l_dict['grasp'] = l_[2]
        l_dict['left'] = l_[3]
        l_dict['right'] = l_[4]
        l_dict['front'] = l_[5]
        l_dict['back'] = l_[6]
        l_dict['red'] = l_[7]
        l_dict['green'] = l_[8]
        l_dict['blue'] = l_[9]
        l_dict['yellow'] = l_[10]
        l_dict['purple'] = l_[11]
        l_dict['stack'] = l_[12]
        l_dict['and'] = l_[13]
        l_dict['put'] = l_[14]
        l_dict['on'] = l_[15]
        l_dict['top'] = l_[16]
        l_dict['of'] = l_[17]
        l_dict['then'] = l_[18]
        l_dict['move'] = l_[19]

        lang_str = ""
        for i in range(len(lang_vec)):
            for k in l_dict:
                if torch.argmax(lang_vec[i]) == torch.argmax(l_dict[k]):
                    lang_str += k + " "
        return lang_str

    def plot_loss_(self, fn, name='fig.png', y_llim=0, y_ulim=1, alpha=0.5, var=False, exep=['kld']):
        loss = []
        for f in range(len(fn)):
            print(fn[f])
            loss_dict = np.load(str(fn[f]))
            loss.append(loss_dict)
        keys = loss_dict.files
        loss_mean = {}
        loss_var = {}
        for key in keys:
            print(key)
            loss_mean[key] = []
            loss_var[key] = []
            for i in range(len(loss_dict['kld'])):
                mean_loss = 0
                sum_loss = 0
                for l in range(len(loss)):
                    sum_loss += loss[l][key][i]
                mean_loss += sum_loss / len(loss)
                loss_mean[key].append(mean_loss)
                sd_ = 0
                var = 0
                for l in range(len(loss)):
                    sd_ += (loss[l][key][i] - mean_loss) ** 2
                var += sd_  # /len(loss)
                loss_var[key].append(var)  # actually sd
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        plt.title(fn)
        x = np.array([i for i in range(len(loss_mean['kld']))])
        new_keys = keys
        for key in keys:
            for e in range(len(exep)):
                if key == exep[e]:
                    new_keys.remove(key)
        for key in new_keys:
            ax.plot(loss_mean[key], label=key)
            ax.fill_between(x, np.array(loss_mean[key]) - np.array(loss_var[key]),
                            np.array(loss_mean[key]) + np.array(loss_var[key]), alpha=alpha)
            ax.set_ylim(y_llim, y_ulim)
        ax.legend()
        plt.show()
        plt.tight_layout()

        plt.savefig(name)

    def gen_markers(self, label, colors):
        len_ = len(label)
        max_len = 5
        col, markers = [], []

        for l in range(len_):
            for i in range(len(colors)):
                c_, _ = dp.lang_vec(colors[i], max_len=max_len)
                if np.argmax(label[l][2]) == np.argmax(c_) or np.argmax(label[l][1]) == np.argmax(c_):
                    if colors[i] == 'red':
                        c = 'r'
                    elif colors[i] == 'green':
                        c = 'g'
                    elif colors[i] == 'blue':
                        c = 'b'
                    elif colors[i] == 'purple':
                        c = 'm'
                    elif colors[i] == 'yellow':
                        c = 'y'
            G_, _ = dp.lang_vec('green', max_len=max_len)
            R_, _ = dp.lang_vec('red', max_len=max_len)
            B_, _ = dp.lang_vec('blue', max_len=max_len)
            P_, _ = dp.lang_vec('purple', max_len=max_len)
            Y_, _ = dp.lang_vec('yellow', max_len=max_len)
            g_, _ = dp.lang_vec('grasp', max_len=max_len)
            m_, _ = dp.lang_vec('move', max_len=max_len)
            l_, _ = dp.lang_vec('left', max_len=max_len)
            r_, _ = dp.lang_vec('right', max_len=max_len)
            f_, _ = dp.lang_vec('front', max_len=max_len)
            b_, _ = dp.lang_vec('back', max_len=max_len)
            p_, _ = dp.lang_vec('put', max_len=max_len)
            o_, _ = dp.lang_vec('on', max_len=max_len)
            task = None

            if np.argmax(label[l][1]) == np.argmax(g_) or np.argmax(label[l][0]) == np.argmax(g_):
                task = 'grasp'
                m = 'o'
            elif np.argmax(label[l][1]) == np.argmax(m_) or np.argmax(label[l][0]) == np.argmax(m_):
                if np.argmax(label[l][3]) == np.argmax(l_) or np.argmax(label[l][2]) == np.argmax(l_):
                    task = 'move left'
                    m = '<'
                elif np.argmax(label[l][3]) == np.argmax(r_) or np.argmax(label[l][2]) == np.argmax(r_):
                    task = 'move right'
                    m = '>'
                elif np.argmax(label[l][3]) == np.argmax(f_) or np.argmax(label[l][2]) == np.argmax(f_):
                    task = 'move front'
                    m = "^"
                elif np.argmax(label[l][3]) == np.argmax(b_) or np.argmax(label[l][2]) == np.argmax(b_):
                    task = 'move back'
                    m = 'v'
            elif np.argmax(label[l][1]) == np.argmax(p_) or np.argmax(label[l][0]) == np.argmax(p_):
                if np.argmax(label[l][3]) == np.argmax(o_) or np.argmax(label[l][2]) == np.argmax(o_):
                    if np.argmax(label[l][3]) == np.argmax(G_) or np.argmax(label[l][4]) == np.argmax(G_):
                        task = 'put on green'
                        m = 'x'
                    elif np.argmax(label[l][3]) == np.argmax(B_) or np.argmax(label[l][4]) == np.argmax(B_):
                        task = 'put on blue'
                        m = 's'
                    elif np.argmax(label[l][3]) == np.argmax(Y_) or np.argmax(label[l][4]) == np.argmax(Y_):
                        task = 'put on yellow'
                        m = 'd'

            # print("task = {}".format(task))
            markers.append(m)
            col.append(c)
        return col, markers


    def loss_curve(self, loss_dict="loss.npz", name='loss.png'):
        """
        generate loss curve from loss.npz
        """
        loss = np.load(loss_dict)
        keys = loss.files
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        for key in keys:
            ax.plot(loss[key], label=key)
        ax.legend()

    def save_txt(self, text, name):
        txt_path = '{}/{}.txt'.format(self.work_dir, name)

        np.savetxt(txt_path, text)

    def save_mot(self, m, name):
        m_path = '{}/{}.npy'.format(self.work_dir, name)
        np.save(m_path, np.array(m))

    def save_b_state(self, b_state, name):
        b_path = '{}/{}.npy'.format(self.work_dir, name)
        np.save(b_path, np.array(b_state))

    def save_arg(self, arg):

        self.session_file = '{}/config.yaml'.format(self.work_dir)

        # save arg
        arg_dict = vars(arg)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        with open(self.session_file, 'w') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)

    def print_log(self, str, print_time=True):
        if print_time:
            # localtime = time.asctime(time.localtime(time.time()))
            str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str

        if self.print_to_screen:
            print(str)
        if self.save_log:
            with open('{}/log.txt'.format(self.work_dir), 'a') as f:
                print(str, file=f)

    def init_timer(self, *name):
        self.record_time()
        self.split_timer = {k: 0.0000001 for k in name}

    def check_time(self, name):
        self.split_timer[name] += self.split_time()

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_timer(self):
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(self.split_timer.values()))))
            for k, v in self.split_timer.items()
        }
        self.print_log('Time consumption:')
        for k in proportion:
            self.print_log(
                '\t[{}][{}]: {:.4f}'.format(k, proportion[k],self.split_timer[k])
                )


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2dict(v):
    return eval('dict({})'.format(v))  #pylint: disable=W0123


def _import_class_0(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

