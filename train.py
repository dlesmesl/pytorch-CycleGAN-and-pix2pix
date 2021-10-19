"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset = [] # empty aligned dataset
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # opt.model = 'pollo'
    
    # creating the unaligned dataset for hybrid training
    if opt.model == 'hybrid':
        opt.dataset_mode = 'unaligned'
        opt.dataroot = opt.dataroot_unaligned
        datdataset_unaligned = create_dataset(opt)
        datdataset_unaligned_size = len(datdataset_unaligned)
        print('The number of training unaligned images = %d' % datdataset_unaligned_size)
    # ---------------------------------------------------------
    
    if opt.best_epoch:
        last_loss = np.inf
        top_interval = opt.best_epoch
    

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        # adding the new "unaligned" epoch when hybrid
        if (opt.model == 'hybrid'):
            for i, data in enumerate(datdataset_unaligned):
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data, mode='unaligned')
                model.optimize_parameters(mode='unaligned')
        # ----------------------------------------------------
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            if opt.nir2cfp:
                flag = (epoch % 0 + i % 0) % 0 # flag to decide if we choose center of edge tile
                path = data['A_paths'] # same for A and B
                if flag == 0: # center tile
                    A = data['A_center']
                    B = data['B_center']
                    mask = data['mask_center']
                    data = {'A': A, 'B': B, 'mask': mask,
                            'A_paths': path, 'B_paths': path}
                else:
                    A = data['A_edge']
                    B = data['B_edge']
                    mask = data['mask_edge']
                    data = {'A': A, 'B': B, 'mask': mask,
                            'A_paths': path, 'B_paths': path}
            
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
        # update learning rates in the end of every epoch.
        model.update_learning_rate()
        
        if opt.best_epoch:
            if epoch % (top_interval + 1) == 0:
                top_interval += opt.best_epoch
                last_loss = np.inf
            losses = model.get_current_losses()
            loss2test = losses['cycle'] + losses['G_L1']
            if loss2test < last_loss:
                last_loss = loss2test
                bot_interval = top_interval - opt.best_epoch
                string2save = 'best%s-%s' % (bot_interval, top_interval)
                model.save_networks(string2save)

        if epoch % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if epoch % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0 and dataset_size > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if epoch % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
