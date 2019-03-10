import os
import datetime
from collections import namedtuple, defaultdict
from shutil import copyfile
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics.regression import mean_squared_error as mse


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import hyperParams as hp



######################## Label ########################################

def get_fgnet_person_loader(root):
    return DataLoader(dataset=ImageFolder(root, transform=pil_to_tensor_transforms), batch_size=1)

def str_to_tensor(text, normalize=False):
    age_group, gender = text.split('.')
    age_tensor = - torch.ones(hp.NUM_AGES)
    age_tensor[int(age_group)] *= -1
    gender_tensor = - torch.ones(hp.NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    if normalize:
        gender_tensor = gender_tensor.repeat(hp.NUM_AGES // hp.NUM_GENDERS)
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result

class Label(namedtuple('Label', ('age', 'gender'))):
    """
    get label
    """
    def __init__(self, age, gender):
        super(Label, self).__init__()
        self.age_group = self.age_transform(self.age)

    def to_str(self):
        return '{}.{}'.format(self.age_group, self.gender)

    @staticmethod
    def age_transform(age):
        age -= 1
        if age < 20:
            # first 4 age groups are for kids <= 20, 5 years intervals
            return max(age // 5, 0)
        else:
            # last (6?) age groups are for adults > 20, 10 years intervals
            return min(4 + (age - 20) // 10, hp.NUM_AGES - 1)

    def to_tensor(self, normalize=False):
        return str_to_tensor(self.to_str(), normalize=normalize)

########################### dataset collection ##############################

pil_to_tensor_transforms = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(2).sub(1))  # Tensor elements domain: [0:1] -> [-1:1]
    ]
)
def sort_to_classes(root, print_cycle=np.inf):
    # Example UTKFace cropped and aligned image file format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
    # Should be 23613 images, use print_cycle >= 1000
    # Make sure you have > 100 MB free space

    def log(text):
        print('[UTKFace dset labeler] ' + text)

    log('Starting labeling process...')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if not files:
        raise FileNotFoundError('No image files in '+root)
    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        matcher = hp.UTKFACE_ORIGINAL_IMAGE_FORMAT.match(f)
        if matcher is None:
            continue
        age, gender, dtime = matcher.groups()
        srcfile = os.path.join(root, f)
        label = Label(int(age), int(gender))
        dstfolder = os.path.join(sorted_folder, label.to_str())
        dstfile = os.path.join(dstfolder, dtime+'.jpg')
        if os.path.isfile(dstfile):
            continue
        if not os.path.isdir(dstfolder):
            os.mkdir(dstfolder)
        copyfile(srcfile, dstfile)
        copied_count += 1
        if copied_count % print_cycle == 0:
            log('Copied %d files.' % copied_count)
    log('Finished labeling process.')


def get_utkface_dataset(root):
    print(root)
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_tensor_transforms)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()


##################### dir and time ########################

fmt_t = "%H_%M"
fmt = "%Y_%m_%d"

def default_train_results_dir():
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))


def default_save_path(eval=True):
    path_str = os.path.join('.', 'results', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))
    if not os.path.exists(path_str):
        os.makedirs(path_str)


def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)


def print_timestamp(s):
    print("[{}] {}".format(datetime.datetime.now().strftime(fmt_t.replace('_', ':')), s))

# LOSS

class LossTracker(object):
    def __init__(self, use_heuristics=False, plot=False, eps=1e-3):
        # assert 'train' in names and 'valid' in names, str(names)
        self.losses = defaultdict(lambda: [])
        self.paths = []
        self.epochs = 0
        self.use_heuristics = use_heuristics
        if plot:
           # print("names[-1] - "+names[-1])
            plt.ion()
            plt.show()
        else:
            plt.switch_backend("agg")

    # deprecated
    def append(self, train_loss, valid_loss, tv_loss, uni_loss, path):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.tv_losses.append(tv_loss)
        self.paths.append(path)
        self.epochs += 1
        if self.use_heuristics and self.epochs >= 2:
            delta_train = self.train_losses[-1] - self.train_losses[-2]
            delta_valid = self.valid_losses[-1] - self.valid_losses[-2]
            if delta_train < -self.eps and delta_valid < -self.eps:
                pass  # good fit, continue training
            elif delta_train < -self.eps and delta_valid > +self.eps:
                pass  # overfit, consider stop the training now
            elif delta_train > +self.eps and delta_valid > +self.eps:
                pass  # underfit, if this is in an advanced epoch, break
            elif delta_train > +self.eps and delta_valid < -self.eps:
                pass  # unknown fit, check your model, optimizers and loss functions
            elif 0 < delta_train < +self.eps and self.epochs >= 3:
                prev_delta_train = self.train_losses[-2] - self.train_losses[-3]
                if 0 < prev_delta_train < +self.eps:
                    pass  # our training loss is increasing but in less than eps,
                    # this is a drift that needs to be caught, consider lower eps next time
            else:
                pass  # saturation \ small fluctuations

    def append_single(self, name, value):
        self.losses[name].append(value)

    def append_many(self, **names):
        for name, value in names.items():
            self.append_single(name, value)

    def append_many_and_plot(self, **names):
        self.append_many(**names)

    def plot(self):
        print("in plot")
        plt.clf()
        graphs = [plt.plot(loss, label=name)[0] for name, loss in self.losses.items()]
        plt.legend(handles=graphs)
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        print("in show")
        plt.show()

    @staticmethod
    def save(path):
        plt.savefig(path, transparent=True)

    def __repr__(self):
        ret = {}
        for name, value in self.losses.items():
            ret[name] = value[-1]
        return str(ret)


def mean(l):
    return np.array(l).mean()


def remove_trained(folder):
    if os.path.isdir(folder):
        removed_ctr = 0
        for tm in os.listdir(folder):
            tm = os.path.join(folder, tm)
            if os.path.splitext(tm)[1] == hp.TRAINED_MODEL_EXT:
                try:
                    os.remove(tm)
                    removed_ctr += 1
                except OSError as e:
                    print("Failed removing {}: {}".format(tm, e))
        if removed_ctr > 0:
            print("Removed {} trained models from {}".format(removed_ctr, folder))


def merge_images(batch1, batch2):
    assert batch1.shape == batch2.shape
    merged = torch.zeros(batch1.size(0) * 2, batch1.size(1), batch1.size(2), batch1.size(3), dtype=batch1.dtype)
    for i, (image1, image2) in enumerate(zip(batch1, batch2)):
        merged[2 * i] = image1
        merged[2 * i + 1] = image2
    return merged

def create_list_of_img_paths(pattern, start, step):
    result = []
    fname = pattern.format(start)
    while os.path.isfile(fname):
        result.append(fname)
        start += step
        fname = pattern.format(start)
    return result

def create_gif(img_paths, dst, start, step):
    BLACK = (255, 255, 255)
    WHITE = (255, 255, 255)
    MAX_LEN = 1024
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    corner = (2, 25)
    fontScale = 0.5
    fontColor = BLACK
    lineType = 2
    for path in img_paths:
        image = cv2.imread(path)
        height, width = image.shape[:2]
        current_max = max(height, width)
        if current_max > MAX_LEN:
            height = int(height / current_max * MAX_LEN)
            width = int(width / current_max * MAX_LEN)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, WHITE)
        cv2.putText(image, 'Epoch: ' + str(start), corner, font, fontScale, fontColor, lineType)
        image = image[..., ::-1]
        frames.append(image)
        start += step
    imageio.mimsave(dst, frames, 'GIF', duration=0.5)
