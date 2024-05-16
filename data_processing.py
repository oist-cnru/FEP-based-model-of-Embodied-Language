import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
import h5py
import torch
import os
import sys

class data_processing():
    def __init__(self, loc_name="folder", index="folder index", save_dir='save folder'):
        self.loc = loc_name
        self.index = index
        self.save_dir = save_dir
        print("preprocessing data ...")
        self.t_max = np.array([20.71693, 61.53293103, 100.63332691000001, 79.78989003, 22.997898969999998, 104.9758021])
        self.joint_min = np.array([-16.73393, 9.03006997, 36.703670089999996, 53.32910897, -13.815899969999998, -3.9638001])
        self.joint_max = np.array(
            [20.71693, 61.53293103, 100.63332691000001, 79.78989003, 22.997898969999998, 102.99508206])

        self.joint_max_TPM = np.array(
            [20.71693, 61.53293103, 100.63332691000001, 79.78989003, 22.997898969999998, 102.99508206])
        self.joint_min_TPM = np.array(
            [-16.73393, 9.03006997, 36.703670089999996, 53.32910897, -13.815899969999998, -1.9830800599999998])

        self.joint_num = 8

        self.joint_indices = np.array([0, 1, 3, 5, 6, 7])

        # convert softmax to jointangle for torbo arm:
        # input : t x (joint_enc_dim*joints) ot : t x joints
        self.joint_enc_dim = 10  # Number of Softmax Units per analog dimension
        self.joints = 6

        self.joint_reference = np.linspace(-1, 1, self.joint_enc_dim)
        # minVal = -0.0; maxVal = 1.0;
        # motor_max = torch.tensor([ 19.657   ,  60.047001,  98.823997,  79.041   ,  21.955999, 100.024002]);
        # motor_min = torch.tensor([-15.674   ,  10.516   ,  38.513   ,  54.077999, -12.774   ,0.988   ]);

        self.joint_reference = np.linspace(-1, 1, self.joint_enc_dim)  # .reshape(1,-1).t()
        self.sigma = 0.05

    def process_vision(self, frame_width, frame_height, fn="path.avi"):
        """
        :param fn:
        :return: save the data as a np array
        """
        ext = os.path.splitext(fn)[-1].lower()
        name = os.path.splitext(os.path.basename(fn))[0]
        if ext == '.mp4' or ext == '.avi':
            cap = cv2.VideoCapture(fn)
            if (cap.isOpened() == False):
                print("error opening video")

            fcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('{}/{}/output_{}_{}x{}.avi'.format(self.loc, self.index, name, frame_height, frame_width), fcc, 20.0,
                                  (frame_width, frame_height))
            frames = []
            while cap.isOpened():
                success, frame = cap.read()  # read the frames of the video
                if success:
                    # print("frame_size: {}".format(frame.shape))
                    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # normalize between -1 and 1
                    new_frame = 2 * ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) - 1
                    new_frames = cv2.resize(new_frame, (frame_width, frame_height))
                    new_frame_s = np.rollaxis(new_frames, 2, 0)
                    print(new_frame_s.shape)
                    frames.append(new_frame_s)
                    new_frames = 255*((new_frames - np.min(new_frames))/(np.max(new_frames) - np.min(new_frames)))
                    cv2.imshow('frame', new_frames.astype(np.uint8))
                    out.write(new_frames.astype(np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            visiondata = np.array(frames)
            # stop frame capture
            cap.release()
            cv2.destroyAllWindows()

        elif ext == ".npy" or ext == ".npz":
            data = np.load(fn)
            if type(data) == dict:
                visiondata = data['vision']
            else:
                new_data = data
                imgs = []
                for i in range(len(new_data)):
                    (h, w) = new_data[i].shape[:2]
                    center = (w / 2, h / 2)
                    # rotate the image by 180 degrees
                    M = cv2.getRotationMatrix2D(center, 180, 1.0)
                    rotated = cv2.warpAffine(new_data[i].astype(np.uint8), M, (w, h))
                    frame = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
                    new_frame = cv2.resize(frame, (frame_width, frame_height))
                    new_frame = (new_frame / np.max(new_frame))
                    imgs.append(new_frame)
                visiondata = np.array(imgs)

        np.save("{}/{}/{}_{}x{}.npy"
                .format(self.loc, self.index, name, frame_width, frame_height), visiondata)

    def make_vision_data(self, n_pos, n_sample, frame_width, frame_height):
        """
        get numpy arrays of vision data
        """
        for j in range(n_sample):
            for k in range(n_pos):
                fn = "{}/{}_{}_vision.avi".format(self.loc, k, j)
                self.process_vision(frame_width, frame_height, fn)

    def lang_vec(self, lang, max_len=20):
        """
        :param lang: take language string as input
        :return: corresponding language one hot vector and mask
        """
        l_ = []
        for i in range(20):
            a = np.zeros(20)
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

        l_mask = np.ones(max_len)
        lang = lang.split(' ')
        lang_vec = [l_dict[i] for i in lang]
        if len(lang_vec) < max_len:
            l_mask[len(lang_vec):] = 0.
            for i in range(max_len - len(lang_vec)):
                lang_vec.append(l_[0])

        return lang_vec, l_mask

    def vec_to_lang(self, lang_vec):
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

    def gen_mask(self, vision, motor, max_len=0):
        """
        :param vision: vision sequence
        :return: corresponding mask for vision and motor sequence assuming that they are of the same length
        """
        #shape of vision = seq_len, c,h,w
        #shape of motor = seq_len, motor_dim
        masks = np.ones(max_len)
        new_vis = np.zeros((max_len, vision.shape[1], vision.shape[2], vision.shape[3]))
        new_mot = np.zeros((max_len, motor.shape[1]))
        if len(vision) < max_len:
            masks[len(vision): ] = 0.
            new_vis[:len(vision), :, :, :] = vision
            new_mot[:len(vision), :] = motor
        return masks, new_vis, new_mot

    def get_max_len(self, n_pos, n_seq, fn, fs):
        """
        go through all sequences and return the length of the longest sequence
        """
        lenv = []
        if fs == 'grasp':
            fs = ''
        for j in range(n_pos):
            for k in range(n_seq):
                fn_m = fn + "{}_{}_motor_{}".format(j, k, fs)
                m = np.loadtxt(fn_m)
                lenv.append(len(m))
        print("maxlen = " + str(np.max(lenv)))
        return np.max(lenv)

    def plot_pos(self, n_seq, n_pos, fn, task='stack'):
        """
        generate object positions from the dataset
        note: still incomplete
        """
        x1, x2, y1, y2 = [], [], [], []
        base_x, base_y = [], []
        left_x, left_y = [], []
        right_x, right_y = [], []
        front_x, front_y = [], []
        back_x, back_y = [], []

        for j in range(n_pos):
            for k in range(n_seq):
                fn_m = fn + "/{}_{}_meta".format(j, k)
                fp = open(fn_m, 'r')
                lines = fp.readlines()
                for line in lines:
                    if task=='stack':
                        if str(line.split('=')[0]) == "base":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            base_x.append(pos[0])
                            base_y.append(pos[1])
                        elif str(line.split('=')[0]) == "mid_start":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            x1.append(pos[0])
                            y1.append(pos[1])
                        elif str(line.split('=')[0]) == "top_start":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            x2.append(pos[0])
                            y2.append(pos[1])
                    elif task=='grasp':
                        if str(line.split('=')[0]) == "goal":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            base_x.append(pos[0])
                            base_y.append(pos[1])
                        if str(line.split('=')[0]) == "left":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            left_x.append(pos[0])
                            left_y.append(pos[1])
                        if str(line.split('=')[0]) == "right":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            right_x.append(pos[0])
                            right_y.append(pos[1])
                        if str(line.split('=')[0]) == "front":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            front_x.append(pos[0])
                            front_y.append(pos[1])
                        if str(line.split('=')[0]) == "back":
                            pos = re.findall(r'[\d]*[.][\d]+', str(line.split('=')[-1]))
                            back_x.append(pos[0])
                            back_y.append(pos[1])
        return base_x, base_y, x1, y1, x2, y2, left_x, left_y, right_x, right_y, front_x, front_y, back_x, back_y

    def scatter_plot2d(self, data, figsize=(10,10), fn="scatter2D.png"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for i in range(len(data)):
            ax.scatter(data[i][0], data[i][1])
        plt.savefig(fn)

    def scatter_plot3d(self, data, figsize=(10, 10), fn="scatter3D.png"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, axes='3d')
        for i in range(len(data)):
            ax.scatter(data[i][0], data[i][1], data[i][2])
        plt.savefig(fn)

    def plot_loss(self, fn, y_ulim=1, save='loss'):
        loss_dict = np.load(fn)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        plt.title(fn)
        for key in loss_dict.files:
            if key != "kld":
                ax.plot(loss_dict[key], label=key)
                ax.set_ylim(0, y_ulim)
        ax.legend()
        plt.savefig(save + fn + '.png')

    def make_training_data(self, fn, fs, language,  n_seq, n_pos, seq_len=110, task='move'):
        """
        make dataset for a particular language and its corresponding behavior
        """
        vision, motor, mask, lang, lang_mask = [], [], [], [], []
        lang_vec, l_mask = self.lang_vec(language, max_len=5)
        # max_len = self.get_max_len(n_pos, n_seq, fn, fs)
        import random
        random.seed(0)
        pg = random.sample(range(20), 10)
        pl = random.sample(range(20), 10)
        pr = random.sample(range(20), 10)
        pf = random.sample(range(20), 10)
        pb = random.sample(range(20), 10)

        for j in range(n_pos):
            for k in range(n_seq):
                if fs == 'grasp':
                    max_len = 330
                    fs = '_'
                    fn_m = fn + "{}_{}_motor{}".format(pg[j], k, fs)
                    fn_v = fn + "grasp/{}_{}_vision_{}64x64.npy".format(j+10, k, fs)
                    fs = 'grasp'
                elif task == 'move':
                    max_len = 330
                    if fs == "left2" or fs == "left":
                        fn_m = fn + "{}_{}_motor_{}".format(pl[j], k, fs)
                        fn_v = fn + "{}/{}_{}_vision_{}_64x64.npy".format(fs, pl[j], k, fs)
                    elif fs == "right2" or fs == "right":
                        fn_m = fn + "{}_{}_motor_{}".format(pr[j], k, fs)
                        fn_v = fn + "{}/{}_{}_vision_{}_64x64.npy".format(fs, pr[j], k, fs)
                    elif fs == "front2" or fs == "front":
                        fn_m = fn + "{}_{}_motor_{}".format(pf[j], k, fs)
                        fn_v = fn + "{}/{}_{}_vision_{}_64x64.npy".format(fs, pf[j], k, fs)
                    elif fs == "back2" or fs == "back":
                        fn_m = fn + "{}_{}_motor_{}".format(pb[j], k, fs)
                        fn_v = fn + "{}/{}_{}_vision_{}_64x64.npy".format(fs, pb[j], k, fs)
                elif task == 'stack2':
                    max_len = 330
                    fn_m = fn + "{}_{}_motor".format(j, k)
                    fn_v = fn + "{}/{}_{}_vision_64x64.npy".format(1, j, k)
                elif task == 'stack3':
                    max_len = 700
                    fn_m = fn + "{}_{}_motor".format(j, k)
                    fn_v = fn + "1/{}_{}_vision_64x64.npy".format(j, k)
                elif task == 'old_stack3':
                    max_len = 700
                    fn_m = fn + "{}_{}_motor".format(j, k)
                    fn_v = fn + "i/{}_{}_vision_64x64.npy".format(j, k)
                fn_l = fn + "{}_{}_meta".format(j, k)
                with open(fn_l) as fp:
                    line = fp.readline()
                    if str(line.strip()) == 'stack=ABC':
                        lang_vec, l_mask = self.lang_vec("stack red green blue .", max_len=5)
                    elif str(line.strip()) == 'stack=ACB':
                        lang_vec, l_mask = self.lang_vec("stack red blue green .", max_len=5)
                    elif str(line.strip()) == 'stack=BAC':
                        lang_vec, l_mask = self.lang_vec("stack green red blue .", max_len=5)
                    elif str(line.strip()) == 'stack=BCA':
                        lang_vec, l_mask = self.lang_vec("stack green blue red .", max_len=5)
                    elif str(line.strip()) == 'stack=CAB':
                        lang_vec, l_mask = self.lang_vec("stack blue red green .", max_len=5)
                    elif str(line.strip()) == 'stack=CBA':
                        lang_vec, l_mask = self.lang_vec("stack blue green red .", max_len=5)

                m = np.loadtxt(fn_m)
                v = np.load(fn_v)
                b_mas = np.ones((len(v)))
                n = int(max_len / seq_len)
                _m = np.zeros((m.shape[0], 6))
                for i in range(len(_m)):
                    _m[i][0] = m[i][0]
                    _m[i][1] = m[i][1]
                    _m[i][2] = m[i][3]
                    _m[i][3] = m[i][5]
                    _m[i][4] = m[i][6]
                    _m[i][5] = m[i][7]
                print("len = " + str(len(v)))
                if len(v) < max_len:
                    v_ = np.concatenate((v, np.stack((np.tile(v[-1], (max_len - len(v), 1, 1, 1))))))
                    m_ = np.concatenate((_m, np.stack((np.tile(_m[-1], (max_len - len(v), 1))))))
                    b_mas_ = np.concatenate((b_mas, np.zeros((max_len - len(v)))))
                else:
                    v_ = v
                    m_ = _m
                    b_mas_ = b_mas

                c, h, w = v.shape[1], v.shape[2], v.shape[3]
                new_v = np.array(v_[::n][:seq_len])
                vision.append(list(new_v))
                mask_ = b_mas_[::n][:seq_len]
                mask.append(list(mask_))
                new_m = m_[::n][:seq_len]
                new_m = (new_m - np.min(new_m)) / (np.max(new_m) - np.min(new_m))  ##normalize
                sc, new_m_, (y, er) = self.convert_to_softmax(new_m)  # TPM for motor
                motor.append(list(new_m_))
                lang.append(list(lang_vec))
                lang_mask.append(list(l_mask))
        save_loc = self.save_dir + language.replace('.', '').replace(' ', '') + "64x64.h5"
        hf5 = h5py.File(save_loc, 'w')
        hf5.create_dataset("motor", data=np.array(motor))
        hf5.create_dataset("language", data=np.array(lang))
        hf5.create_dataset("lang_mask", data=np.array(lang_mask))
        hf5.create_dataset("vision", data=np.array(vision))
        hf5.create_dataset("mask", data=np.array(mask))
        hf5.close()
        return str(save_loc)

    def combine_datasets(self, fns=['h5 files of datasets to combine'], fn="new_dataset.h5", tasks=None, n_seq=10):
        """
        function to concatenate datasets with same dimension for vision, motor, language and corresponding masks
        need to verify shapes before concatenating
        """
        data=[]
        new_data = {'motor': [], 'vision': [], 'language': [], 'mask': [], 'lang_mask': []}

        keys = new_data.keys()
        n_datasets = len(fns)
        print("n_datasets to be combines = " + str(n_datasets))
        for i in range(n_datasets):
            data.append(h5py.File(fns[i], 'r'))
        # for i in range(n_data):
        #     data.append(h5py.File(fns[i], 'r'))
        for i in range(len(data)):
            for q in keys:
                if tasks == 'comb':
                    new_data[q] += list(data[i][q])
                elif tasks == 'pos_train':
                    new_data[q] += list(data[i][q][:n_seq])
                elif tasks == 'pos_test':
                    new_data[q] += list(data[i][q][n_seq:])
                else:
                    new_data[q] += list(data[i][q][:n_seq])
        hf = h5py.File(fn, 'w')
        for k in new_data.keys():
            print("{} shape = {}, type = {}".format(k, np.array(new_data[k]).shape, type(new_data[k])))
            hf.create_dataset(k, data=np.array(new_data[k]))
        hf.close()

    def encode_softmax(self, softmax_config, val):
        val_softmax = None
        if isinstance(val, torch.Tensor):
            val_softmax = val.new_empty((val.shape[0] * softmax_config['num_basis_functions']), dtype=val.dtype)
            for dim in range(val.shape[0]):
                offset = dim * softmax_config['num_basis_functions']
                val_softmax[offset:offset + softmax_config['num_basis_functions']] = torch.softmax(
                    -((softmax_config['centers'][:, dim] - val[dim]) ** 2) / softmax_config['sigma'][dim], dim=0)
        elif isinstance(val, np.ndarray):
            val_softmax = np.empty((val.shape[0] * softmax_config['num_basis_functions']))
            for dim in range(val.shape[0]):
                offset = dim * softmax_config['num_basis_functions']
                # print("---")
                # print(softmax_config['centers'][:,dim]-val[dim])
                # print(softmax_config['centers'][:,dim])
                # print(val[dim])
                val_softmax[offset:offset + softmax_config['num_basis_functions']] = np.exp(
                    -((softmax_config['centers'][:, dim] - val[dim]) ** 2) / softmax_config['sigmas'][dim])
                val_softmax[offset:offset + softmax_config['num_basis_functions']] /= np.sum(
                    val_softmax[offset:offset + softmax_config['num_basis_functions']])
        else:
            raise ValueError('Unknown input type %s.' % (type(val)))
        return val_softmax
    def decode_softmax(self, softmax_config, val):
        val_decoded = None
        if isinstance(val, torch.Tensor):
            val_decoded = val.new_empty(int((val.shape[0] / softmax_config['num_basis_functions'])), dtype=val.dtype)
            for dim in range(val_decoded.shape[0]):
                offset = dim * softmax_config['num_basis_functions']
                val_decoded[dim] = np.inner(val[offset:offset + softmax_config['num_basis_functions']],
                                            softmax_config['centers'][:, dim])
                # todo torch.dot not working here ?!
        elif isinstance(val, np.ndarray):
            val_decoded = np.empty(int((val.shape[0] / softmax_config['num_basis_functions'])))
            for dim in range(val_decoded.shape[0]):
                offset = dim * softmax_config['num_basis_functions']
                val_decoded[dim] = np.inner(val[offset:offset + softmax_config['num_basis_functions']],
                                            softmax_config['centers'][:, dim])
        else:
            raise ValueError('Unknown input type %s.' % (type(val)))
        return val_decoded
    def normalize_data(self, indata, data_offset=0.05, dat_min=[], dat_max=[]):
        num_dims = len(indata.shape)
        num_inlen = indata.shape[-1]
        inshape = indata.shape
        outshape = list(inshape)
        s = indata.shape
        indata_flat = indata.reshape((np.prod(s[:-1]), s[-1]))
        outdata = np.zeros(outshape)
        s = outdata.shape
        outdata_flat = outdata.reshape((np.prod(s[:-1]), s[-1]))
        if len(dat_min) == 0:
            dat_min = indata_flat.min(axis=0)
        if len(dat_max) == 0:
            dat_max = indata_flat.max(axis=0)
        dat_dist = np.array(dat_max) - np.array(dat_min)
        print(dat_dist)
        dat_offset = data_offset * dat_dist
        dat_min = dat_min - dat_offset
        dat_max = dat_max + dat_offset
        dat_dist = dat_max - dat_min
        outdata_flat[:, :] = (indata_flat[:, :] - dat_min[np.newaxis, :]) / dat_dist[np.newaxis, :]
        norm_data = {'dat_min': dat_min, 'dat_max': dat_max, 'dat_dist': dat_dist}
        return norm_data, outdata
    def inv_normalize_data(self, datacfg, indata):
        num_dims = len(indata.shape)
        num_inlen = indata.shape[-1]
        inshape = indata.shape
        outshape = list(inshape)
        s = indata.shape
        indata_flat = indata.reshape((np.prod(s[:-1]), s[-1]))
        outdata = np.zeros(outshape)
        s = outdata.shape
        outdata_flat = outdata.reshape((np.prod(s[:-1]), s[-1]))
        dat_min = datacfg['dat_min']
        dat_max = datacfg['dat_max']
        dat_dist = datacfg['dat_dist']
        # outdata_flat[:,:]=(indata_flat[:,:]-dat_min[np.newaxis,:])/dat_dist[np.newaxis,:]
        # inv:
        outdata_flat[:, :] = indata_flat[:, :] * dat_dist[np.newaxis, :] + dat_min[np.newaxis, :]
        return outdata
    def convert_to_softmax(self, indata, num_basis_functions=10, data_range=(0, 1), basis_function_sigma=0.01):
        num_dims = len(indata.shape)
        num_inlen = indata.shape[-1]
        num_outlen = int(num_inlen * num_basis_functions)
        inshape = indata.shape
        outshape = list(inshape)
        outshape[-1] = num_outlen
        s = indata.shape
        indata_flat = indata.reshape((np.prod(s[:-1]), s[-1]))
        outdata = np.zeros(outshape)
        s = outdata.shape
        outdata_flat = outdata.reshape((np.prod(s[:-1]), s[-1]))
        outdata_reconstruct = np.zeros(inshape)
        s = outdata_reconstruct.shape
        outdata_reconstruct_flat = outdata_reconstruct.reshape((np.prod(s[:-1]), s[-1]))
        if len(data_range) < 1:
            # recalculate range
            print("recalc data range!")
            dat_min = indata_flat.min(axis=0)
            dat_max = indata_flat.max(axis=0)
            dat_dist = dat_max - dat_min
            dat_offset = 0.1 * dat_dist
            dat_min = dat_min - dat_offset
            dat_max = dat_max + dat_offset
            dat_dist = dat_max - dat_min
            data_range = np.vstack((dat_min, dat_max))
        else:
            dat_min = np.array((data_range[0],) * num_inlen)
            dat_max = np.array((data_range[1],) * num_inlen)
            data_range = np.vstack((dat_min, dat_max))
            dat_dist = dat_max - dat_min
        softmax_config = {}
        softmax_config['data_range'] = data_range
        softmax_config['centers'] = np.linspace(data_range[0], data_range[1], num=num_basis_functions)
        softmax_config['sigmas'] = dat_dist * basis_function_sigma
        softmax_config['num_basis_functions'] = num_basis_functions
        err_inputs_sample = 0
        for t in range(indata_flat.shape[0]):
            outdata_flat[t, :] = self.encode_softmax(softmax_config, indata_flat[t, :])
            outdata_reconstruct_flat[t, :] = self.decode_softmax(softmax_config, outdata_flat[t, :])
            err_inputs_sample += np.sum((outdata_reconstruct_flat[t, :] - indata_flat[t, :]) ** 2)
        err_targets = err_inputs_sample / indata_flat.shape[0]
        return softmax_config, outdata, (outdata_reconstruct, err_targets)
    def convert_from_softmax(self, softmax_config, indata):
        num_dims = len(indata.shape)
        num_inlen = indata.shape[-1]
        num_outlen = num_inlen // softmax_config['num_basis_functions']  # //-floor division
        inshape = indata.shape
        outshape = list(inshape)
        outshape[-1] = num_outlen
        s = indata.shape
        indata_flat = indata.reshape((np.prod(s[:-1]), s[-1]))
        outdata = np.zeros(outshape)
        s = outdata.shape
        outdata_flat = outdata.reshape((np.prod(s[:-1]), s[-1]))
        for t in range(indata_flat.shape[0]):
            outdata_flat[t, :] = self.decode_softmax(softmax_config, indata_flat[t, :])
        return outdata

    def TPM(self, indata):
        selected_joints = [indata[idx] for idx in [0,1,3,5,6,7]]
        sigma = 0.05
        selected_joints = np.array(selected_joints)
        joint_reference1 = np.linspace(-1, 1, 6)  # .reshape(1,-1).t()
        normalized_joints = (selected_joints - self.joint_min_TPM) / (self.joint_max_TPM - self.joint_min_TPM)
        normalized_joints = normalized_joints * 2 - 1
        e = np.exp(-((normalized_joints[:, None] - joint_reference1[None, :].T) ** 2 / sigma))
        TPM_joints = e / e.sum(axis=1, keepdims=True)

        return TPM_joints
    def renormalize_mot(self, indata):
        outdata = np.zeros((indata.shape[0], self.joints))
        t_indata = indata  # torch.exp(indata)
        jointnr = 0
        for i in range(0,self.joints):
            outdata[:, jointnr] = t_indata[:, i]  # /np.sum(joint_reference)
            outdata[:, jointnr] = (outdata[:, jointnr] + 1) / 2
            outdata[:, jointnr] = self.joint_min[jointnr] + (outdata[:, jointnr] * (self.joint_max[jointnr] - self.joint_min[jointnr]))
            jointnr = jointnr + 1

        return outdata
    def inv_TPM(self, indata):
        outdata = np.zeros((indata.shape[0], self.joints))
        t_indata = indata  # torch.exp(indata)
        jointnr = 0
        for i in range(0, self.joint_enc_dim * self.joints, self.joint_enc_dim):
            outdata[:, jointnr] = np.matmul(t_indata[:, i:i + self.joint_enc_dim], self.joint_reference)  # /np.sum(joint_reference)
            outdata[:, jointnr] = (outdata[:, jointnr] + 1) / 2
            outdata[:, jointnr] = self.joint_min[jointnr] + (outdata[:, jointnr] * (self.joint_max[jointnr] - self.joint_min[jointnr]))
            jointnr = jointnr + 1

        return outdata
