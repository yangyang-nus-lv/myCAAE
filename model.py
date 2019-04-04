import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import grad

import logging
from tensorboardX import SummaryWriter
import random
from collections import OrderedDict
import cv2
import imageio
from PIL import Image

import hyperParams as hp
from utils import *


class Encoder(nn.Module):
    """
    Encoder E
    Input: 3 * 128 * 128 (C * H * W)
    conv_1: 3 * 128 * 128 -> 64 * 64 * 64 relu
    conv_2: 64 * 64 * 64 -> 128 * 32 * 32 relu
    conv_3: 128 * 32 * 32 -> 256 * 16 * 16 relu
    conv_4: 256 * 16 * 16 -> 512 * 8 * 8 relu
    CONV_5: 512 * 8 * 8 -> 1024 * 4 * 4
    fc: view(-1, 1024 * 4 * 4) -> 50 * 1 tanh
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # conv = nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=1, padding=0, groups=1, bias=True)
        self.conv_1 = nn.Conv2d(3, 64, 5, 2, 2)
        self.conv_2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv_3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv_4 = nn.Conv2d(256, 512, 5, 2, 2)
        self.conv_5 = nn.Conv2d(512, 1024, 5, 2, 2)
        self.fc = nn.Linear(hp.NUM_FC_CHANNELS, hp.LENGTH_Z)

    def forward(self, x):           # x is a face image
        # 5 conv layers
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = self.relu(self.conv_4(x))
        x = self.relu(self.conv_5(x))
        x = x.flatten(1, -1) # flatten batch image from the C dim to 
        # fc layer
        x = self.fc(x)

        return self.tanh(x)

class Generator(nn.Module):
    """
    Generator G
    Input: (50 + 10 + 10) Z + AGE + Gender
    fc: 70 -> 1024 * 4 * 4 relu
    convT_1: 1024 * 4 * 4 -> 512 * 8 * 8
    convT_2: 512 * 8 * 8 -> 256 * 16 * 16
    convT_3: 256 * 16 * 16 -> 128 * 32 * 32
    convT_4: 128 * 32 * 32 -> 64 * 64 * 64
    convT_5: 64 * 64 * 64 -> 32 * 128 * 128
    convT_6: 32 * 128 * 128 -> 16 * 128 * 128
    convT_7: 16 * 128 * 128 -> 3 * 128 * 128
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # fc
        self.fc = nn.Linear(hp.LENGTH_Z + hp.LENGTH_L, hp.NUM_FC_CHANNELS)
        # convTranspose layer
        self.convT_1 = nn.ConvTranspose2d(1024, 512, 5, 2, padding=2, output_padding=1)
        self.convT_2 = nn.ConvTranspose2d(512, 256, 5, 2, padding=2, output_padding=1)
        self.convT_3 = nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1)
        self.convT_4 = nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1)
        self.convT_5 = nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1)
        self.convT_6 = nn.ConvTranspose2d(32, 16, 5, 1, padding=2)
        self.convT_7 = nn.ConvTranspose2d(16, 3, 1, 1)

    def forward(self, z, age=None, gender=None):           # x is a vector
        # fc
        if age and gender:
            label = Label(age, gender).to_tensor() \
                if (isinstance(age, int) and isinstance(gender, int)) \
                else torch.cat((age, gender), 1)
            z = torch.cat((z, label), 1)  # z_l

        z = self.relu(self.fc(z))
        # rshape to 1024 * 4 * 4
        z = z.view(-1, 1024, hp.SIZE_MINI_MAP, hp.SIZE_MINI_MAP)
        # convT
        z = self.relu(self.convT_1(z))
        z = self.relu(self.convT_2(z))
        z = self.relu(self.convT_3(z))
        z = self.relu(self.convT_4(z))
        z = self.relu(self.convT_5(z))
        z = self.relu(self.convT_6(z))
        z = self.tanh(self.convT_7(z))

        return z

class DiscriminatorZ(nn.Module):
    """
    DiscriminatorZ Dz
    Input: (50 + 10 + 10)
    fc_1: 70 -> 64
    fc_2: 64 -> 32
    fc_3: 32 -> 16
    fc_4: 16 -> 1
    """
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        self.relu = nn.LeakyReLU()
        self.fc_1 = nn.Sequential(nn.Linear(hp.LENGTH_Z, hp.NUM_ENCODER_CHANNELS), 
                                nn.BatchNorm1d(hp.NUM_ENCODER_CHANNELS))
        self.fc_2 = nn.Sequential(nn.Linear(hp.NUM_ENCODER_CHANNELS, hp.NUM_ENCODER_CHANNELS // 2),
                                nn.BatchNorm1d(hp.NUM_ENCODER_CHANNELS // 2))
        self.fc_3 = nn.Sequential(nn.Linear(hp.NUM_ENCODER_CHANNELS // 2, hp.NUM_ENCODER_CHANNELS // 4),
                                nn.BatchNorm1d(hp.NUM_ENCODER_CHANNELS // 4))
        self.fc_4 = nn.Linear(hp.NUM_ENCODER_CHANNELS // 4, 1)

    def forward(self, z):
        z = self.relu(self.fc_1(z))
        z = self.relu(self.fc_2(z))
        z = self.relu(self.fc_3(z))
        z = self.fc_4(z)

        return z

class DiscriminatorImg(nn.Module):
    """
    DiscriminatorImg Dimg
    conv_1: 3 * 128 * 128 -> 16 * 64 * 64 InstanceNorm2d + relu
    catenate conv_1(img) and labels -> (16 + 20) * 64 * 64
    conv_2: (16 + 20) * 64 * 64 -> 32 * 32 * 32 InstanceNorm2d + relu
    conv_3: 32 * 32 * 32 -> 64 * 16 * 16 InstanceNorm2d + relu
    conv_4: 64 * 16 * 16 -> 128 * 8 * 8 InstanceNorm2d + relu
    fc_1: 128 * 8 * 8 -> 1024 relu
    fc_2: 1024 -> 1
    """
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 2, 2),
            nn.InstanceNorm2d(16),
            self.relu   # should add () ???
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16 + hp.LENGTH_L, 32, 2, 2),
            nn.InstanceNorm2d(32),
            self.relu   
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.InstanceNorm2d(64),
            self.relu   
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.InstanceNorm2d(128),
            self.relu   
        )
        self.fc_1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc_2 = nn.Linear(1024, 1)

    def forward(self, x, labels, device):
        # conv
        x = self.conv_1(x)
        # reshape label and catenate img and label
        labels_tensor = torch.zeros(torch.Size((x.size(0), labels.size(1), x.size(2), x.size(3))), device=device)
        for img_idx in range(x.size(0)):
            for label in range(labels.size(1)):
                labels_tensor[img_idx, label, :, :] = labels[img_idx, label]  # fill a square
        x = torch.cat((x, labels_tensor), 1)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        # flatten
        x = x.flatten(1, -1)
        # fc
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)

        return x

class CAAE(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.Dimg = DiscriminatorImg()
        self.G = Generator()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = None

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])
        
    def train(
            self,
            utkface_path,
            batch_size=128,
            epochs=128,
            weight_decay=1e-5,
            lr=2e-4,
            should_plot=False,
            betas=(0.9, 0.999),
            valid_size=None,
            save_path=None,
            models_saving='always',):
            
        save_path = save_path or default_save_path()
        
        dataset = get_utkface_dataset(utkface_path)
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(dataset, (valid_size, len(dataset) - valid_size))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        input_output_loss = nn.L1Loss()
        bce_with_logits_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        nrow = round((2 * batch_size)**0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val:
                    optimizer.param_groups[0][param] = val

        # loss_tracker = LossTracker(plot=should_plot)
        save_path_epoch = ""
        # save_count = 0
        paths_for_gif = []

        loss_history = []
        loss_writer = SummaryWriter('runs/wgan_step2')

        for epoch in range(1, epochs + 1):
            save_path_epoch = os.path.join(save_path, "epoch" + str(epoch))
            try:
                if not os.path.exists(save_path_epoch):
                    os.makedirs(save_path_epoch)
                paths_for_gif.append(save_path_epoch)

                # ************************************* loss functions *******************************************************
                # losses = defaultdict(lambda: [])
                self.training_mode()  # move to train mode

                loss_weight = hp.loss_weights(epoch)
                for i, (images, labels) in enumerate(train_loader, 1):

                    images = images.to(device=self.device)
                    labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])  # todo - can remove list() ?
                    labels = labels.to(device=self.device)
                    # print ("DEBUG: iteration: "+str(i)+" images shape: "+str(images.shape))
                    z = self.E(images)
                    
                    # loss function of encoder + generator
                    z_l = torch.cat((z, labels), 1)
                    generated = self.G(z_l)
                    eg_loss = input_output_loss(generated, images) 
                    # losses['eg'].append(eg_loss.item())
                    # loss_writer.add_scalar('eg', eg_loss.item(), epoch)
                    # total variation to smooth the generated image
                    tv_loss = (
                        mse_loss(generated[:, :, :, :-1], generated[:, :, :, 1:]) +\
                        mse_loss(generated[:, :, :-1, :], generated[:, :, 1:, :])
                    )
                    tv_loss.to(self.device)
                    # losses['tv'].append(tv_loss.item())
                    # loss_writer.add_scalar('tv', tv_loss.item(), epoch)

                    # DiscriminatorZ Loss
                    z_prior = (torch.rand_like(z, device=self.device) - 0.5) * 2 # [-1 : 1]
                    d_z_prior_logits = self.Dz(z_prior)
                    d_z_logits = self.Dz(z)

                    dz_loss_prior = bce_with_logits_loss(d_z_prior_logits, torch.ones_like(d_z_prior_logits))
                    dz_loss_z = bce_with_logits_loss(d_z_logits, torch.zeros_like(d_z_logits))
                    # ######################## DiscriminatorZ gradient penalty ##########################
                    # # Calculate interpolation
                    # alpha_z = torch.rand(z.shape[0], 1, device=self.device)

                    # # alpha_z = alpha_z.expand_as(z)
                    # z_inter = alpha_z * z_prior.detach() + (1 - alpha_z) * z.detach()
                    # z_inter.requires_grad = True

                    # # Calculate probability of interpolated examples
                    # prob_z_inter = self.Dz(z_inter)

                    # # Calculate gradients of probabilities with respect to examples
                    # dz_gradients = grad(outputs=prob_z_inter, inputs=z_inter,
                    #                     grad_outputs=torch.ones(prob_z_inter.size(), device=self.device),
                    #                     create_graph=True, retain_graph=True)[0]

                    # # Gradients have shape (batch_size, z_length),
                    # # so flatten to easily take norm per example in batch
                    # # dz_gradients = dz_gradients.view(self.batch_size, -1)
                    # losses['dz_gn'].append(dz_gradients.norm(2, dim=1).mean().item())

                    # # Derivatives of the gradient close to 0 can cause problems because of
                    # # the square root, so manually calculate norm and add epsilon
                    # dz_gradients_norm = torch.sqrt(torch.sum(dz_gradients ** 2, dim=1) + 1e-12)

                    # # Return gradient penalty
                    # dz_gp = loss_weight['dz_gp'] * ((dz_gradients_norm - 1) ** 2).mean()
                    # ##################################################################################
                    dz_loss_tot = dz_loss_z + dz_loss_prior
                    # losses['dz_r'].append(dz_loss_prior.item())
                    # losses['dz_f'].append(dz_loss_z.item())
                    # losses['dz'].append(dz_loss_tot.item())
                    

                    # Encoder\DiscriminatorZ Loss
                    ed_loss = bce_with_logits_loss(d_z_logits, torch.ones_like(d_z_logits))
                    # losses['ed'].append(ed_loss.item())
                    loss_writer.add_scalars('dz', {'dz_r': dz_loss_prior.item(), 'dz_f': dz_loss_z.item(), 'dz': dz_loss_tot.item(), 'ed': ed_loss.item()}, epoch)
                    # DiscriminatorImg Loss
                    d_i_input_logits = self.Dimg(images, labels, self.device)
                    d_i_output_logits = self.Dimg(generated, labels, self.device)

                    di_input_loss = - d_i_input_logits.mean()
                    di_output_loss = d_i_output_logits.mean()
                    ################# DiscriminatorImg gradient penalty ###############################
                    # di_gp = self._di_gradient_penalty(images, generated, labels)
                    # Calculate interpolation
                    alpha_i = torch.rand(images.shape[0], 1, 1, 1, device=self.device)
                    # alpha_i = alpha_i.expand_as(images)
                    interpolated = alpha_i * images.detach() + (1 - alpha_i) * generated.detach()
                    interpolated.requires_grad = True

                    # Calculate probability of interpolated examples
                    prob_interpolated = self.Dimg(interpolated, labels, self.device)

                    # Calculate gradients of probabilities with respect to examples
                    di_gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                                        grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
                                        create_graph=True, retain_graph=True)[0]

                    # Gradients have shape (batch_size, num_channels, img_width, img_height),
                    # so flatten to easily take norm per example in batch
                    di_gradients = di_gradients.view(batch_size, -1)
                    # losses['di_gn'].append(di_gradients.norm(2, dim=1).mean().item())

                    # Derivatives of the gradient close to 0 can cause problems because of
                    # the square root, so manually calculate norm and add epsilon
                    di_gradients_norm = torch.sqrt(torch.sum(di_gradients ** 2, dim=1) + 1e-12)

                    # Return gradient penalty
                    di_gp = loss_weight['di_gp'] * ((di_gradients_norm - 1) ** 2).mean()    
                    ###################################################################################
                    di_loss_tot = di_input_loss + di_output_loss + di_gp
                    
                    # losses['di_r'].append(di_input_loss.item())
                    # losses['di_f'].append(di_output_loss.item())
                    # losses['di_gp'].append(di_gp.item())
                    # losses['di'].append(di_loss_tot.item())

                    loss_writer.add_scalars('di_gp', {'di_gn': di_gradients.norm(2, dim=1).mean().item(), 
                                                    'di_gp': di_gp.item()}, epoch)
                    
                    # Generator\DiscriminatorImg Loss
                    gd_loss = - d_i_output_logits.mean()
                    # losses['gd'].append(gd_loss.item())
                    loss_writer.add_scalars('di', {'di_r': di_input_loss.item(), 'di_f': di_output_loss.item(), 'di': di_loss_tot.item(), 'gd': gd_loss.item()}, epoch)

                    loss = loss_weight['eg'] * eg_loss + loss_weight['tv'] * tv_loss + loss_weight['ed'] * ed_loss + loss_weight['gd'] * gd_loss
                    
                    loss_writer.add_scalars('eg', {'tot': loss.item(), 'eg': eg_loss.item(), 'tv': tv_loss.item()}, epoch)
                    # ************************************* loss functions end *******************************************************

                    # Start back propagation

                    # Back prop on Encoder\Generator
                    self.eg_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.eg_optimizer.step()

                    # Back prop on DiscriminatorZ
                    self.dz_optimizer.zero_grad()
                    dz_loss_tot.backward(retain_graph=True)
                    self.dz_optimizer.step()

                    # Back prop on DiscriminatorImg
                    self.di_optimizer.zero_grad()
                    di_loss_tot.backward()
                    self.di_optimizer.step()

                    now = datetime.datetime.now()

                logging.info('[{h}:{m}] [Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")

                to_save_models = models_saving in ('always', 'tail')
                cp_path = self.save(save_path_epoch, to_save_models=to_save_models)
                if models_saving == 'tail':
                    prev_folder = os.path.join(save_path, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                # loss_tracker.save(os.path.join(cp_path, 'losses.png'))

                with torch.no_grad():  # validation
                    self.eval()  # move to eval mode

                    for val_images, val_labels in valid_loader:
                        val_images = val_images.to(self.device)
                        validate_labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(val_labels.numpy())])
                        validate_labels = validate_labels.to(self.device)

                        z = self.E(val_images)
                        z_l = torch.cat((z, validate_labels), 1)
                        generated = self.G(z_l)

                        validate_loss = input_output_loss(val_images, generated)

                        joined = merge_images(val_images, generated)  
                        file_name = os.path.join(save_path_epoch, 'validation.png')
                        save_image_normalized(tensor=joined, filename=file_name, nrow=nrow)
                        # test first 10 valid images
                        for i in range(10):
                            test_image = val_images[i, :, :, :]
                            test_label = val_labels[i]
                            age, gender = idx_to_class_info(test_label)
                            tested = self.test_single(test_image, age, gender, target=None, save_test=False)
                            if i == 0:
                                test_joined = tested.clone().detach()
                            else:
                                test_joined = torch.cat((test_joined, tested), 0)
                        test_file_name = os.path.join(save_path_epoch, 'test.png')
                        save_image_normalized(tensor=test_joined, filename=test_file_name, nrow=11)

                        # losses['valid'].append(validate_loss.item())
                        break


                # loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
                # loss_tracker.plot()
                loss_writer.add_scalars('tr_va', {'train': eg_loss.item(), 'valid': validate_loss.item()}, epoch)
                # logging.info('[{h}:{m}] [Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))
                # loss_history.append(losses)

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(save_path_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(save_path, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                # loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(save_path_epoch, to_save_models=True)
        # loss_tracker.plot()

        with open('loss_history.txt', 'w') as f:
            for item in loss_history:
                f.write("%s\n" % item)

    def test_single(self, image_tensor, age, gender, target, save_test=True):
        """
            test single image
        """
        self.eval()
        batch = image_tensor.repeat(hp.NUM_AGES, 1, 1, 1).to(device=self.device)  # N x C x H x W
        z = self.E(batch)  # N x Z

        gender_tensor = -torch.ones(hp.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        gender_tensor = gender_tensor.repeat(hp.NUM_AGES, hp.NUM_AGES // hp.NUM_GENDERS)  # apply gender on all images

        age_tensor = -torch.ones(hp.NUM_AGES, hp.NUM_AGES)
        for i in range(hp.NUM_AGES):
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        l = torch.cat((age_tensor, gender_tensor), 1).to(self.device)
        z_l = torch.cat((z, l), 1)

        generated = self.G(z_l)

        joined = torch.cat((image_tensor.unsqueeze(0), generated), 0)

        joined = nn.ConstantPad2d(padding=4, value=-1)(joined)
        for img_idx in (0, Label.age_transform(age) + 1):
            for elem_idx in (0, 1, 2, 3, -4, -3, -2, -1):
                joined[img_idx, :, elem_idx, :] = 1  # color border white
                joined[img_idx, :, :, elem_idx] = 1  # color border white
        if save_test:
            dest = os.path.join(target, 'menifa.png')
            save_image_normalized(tensor=joined, filename=dest, nrow=joined.size(0))
            print_timestamp("Saved test result to " + dest)
            return dest
        else:
            return joined
    
    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.

        :return:
        """

        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name) # getattr(x, 'foobar') is equivalent to x.foobar.
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn('eval')

    def training_mode(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn('training_mode')

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.
        :return: path
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, hp.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            print_timestamp("Saved {} to {}".format(', '.join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.
        :return:
        """
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, hp.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname)())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))
