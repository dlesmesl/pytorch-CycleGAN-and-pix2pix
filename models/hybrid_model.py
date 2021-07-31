import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class HybridModel(BaseModel):
    def name(self):
        return 'HybridModel'
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(pool_size=10, norm='instance')
        parser.set_defaults(no_dropout=True)
        
        if is_train:
            parser.add_argument('--lambda_L1', type=float,
                                default=150.0, help='weight for Paired L1 loss')
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float,
                                default=10.0, help='weight for identity loss')
            
            # Additional arguments (unpaired data)
            parser.add_argument('--dataroot_unaligned', required=True,
                                help='path to unaligned images (should have subfolders trainA, trainB, valA, valB, etc)')
            parser.add_argument('--pool_size_unaligned', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')

        return parser

    def __init__(self, opt):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_A'] # could add more later
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        
        # The naming is different from those used in the paper. (CycleGAN)
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain: # Discriminators (have new use_sigmoid parameter check in networks.py)
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # define buffers to store previously generated images
            self.fake_B_pool_aligned = ImagePool(opt.pool_size)
            self.fake_A_pool_aligned = ImagePool(opt.pool_size)
            
            self.fake_B_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            self.fake_A_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            
            # define buffers to store real images
            self.real_A_pool_aligned = ImagePool(opt.pool_size)
            self.real_B_pool_aligned = ImagePool(opt.pool_size)
            
            self.real_A_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            self.real_B_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionGAN = networks.GANLoss('vanilla').to(
                self.device)  # nn.BCEWithLogitsLoss() is the loss used in ws-i2i
            self.criterionL1 = torch.nn.L1Loss() # Paired loss
            self.criterionIdt = torch.nn.L1Loss() # Identity loss
            self.criterionCycle = torch.nn.L1Loss() # Cycle loss
            
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.lambda_L1 = self.opt.lambda_L1
            
            self.optimizers = []
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
            # ------------important---------------
            # Will skip temporary in lines 79 - 81
            # self.loss_D_real = 0.0
            # self.loss_D_fake = 0.0
            # ------------------------------------
             
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.real_A = input['A' if AtoB else 'B'].to(
            self.device)  # get image data A
        self.real_B = input['B' if AtoB else 'A'].to(
            self.device)  # get image data B
        # get image paths
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, mode='aligned'):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.idt_A = self.netG_A(self.real_B)  # G_A(B) identity?

        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.idt_B = self.netG_B(self.real_A)  # G_B(A) identity?

        #if(mode == 'unaligned'):
        with torch.no_grad():  # CycleGAN uses grad
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
            
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Here the definition might change because the discriminator is relativistic
        pred_real = netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        if(self.opt.loss_case == 'rel_bce'):
            loss_D = self.criterionGAN(pred_real - pred_fake, True)
        else:
            loss_D = (torch.mean((pred_real - torch.mean(pred_fake) - 1.0) ** 2) +
                      torch.mean((pred_fake - torch.mean(pred_real) + 1.0) ** 2))/2
        loss_D.backward()
        return loss_D
    
    # different configuration because it's hybrid
    def backward_D_A(self, mode='aligned'):
        """Calculate GAN loss for discriminator D_A"""
        if mode == 'aligned':
            fake_B = self.fake_B_pool_aligned.query(self.fake_B)
        elif mode == 'unaligned':
            fake_B = self.fake_B_pool_unaligned.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        
    def backward_D_B(self, mode='aligned'):
        """Calculate GAN loss for discriminator D_B"""
        if mode == 'aligned':
            fake_A = self.fake_A_pool_aligned.query(self.fake_A)
        elif mode == 'unaligned':
            fake_A = self.fake_A_pool_unaligned.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        
    # Most complex difference
    def backward_G(self, mode='aligned'):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_L1 = self.lambda_L1
        
        # Selecting the data to use to calculate the relativistic discriminators
        if mode == 'aligned':
            real_B = self.real_B_pool_aligned.query(self.real_B)
            real_A = self.real_A_pool_aligned.query(self.real_A)
        elif mode == 'unaligned':
            real_B = self.real_B_pool_unaligned.query(self.real_B)
            real_A = self.real_A_pool_unaligned.query(self.real_A)
        
        # Discriminating images    
        pred_fake_B = self.netD_A(self.fake_B)
        pred_real_B = self.netD_A(real_B)

        pred_fake_A = self.netD_B(self.fake_A)
        pred_real_A = self.netD_B(real_A)
        
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(
            self.idt_A, self.real_B) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(
            self.idt_B, self.real_A) * lambda_A * lambda_idt
        
        self.loss_G_idt = self.loss_idt_A + self.loss_idt_B
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * lambda_B
        
        self.loss_cycle = self.loss_cycle_A + self.loss_cycle_B
        
        # Relativistic discriminator loss
        if self.opt.loss_case == 'rel_bce':
            self.loss_G_GAN_A = self.criterionGAN(
                pred_fake_B - pred_real_B, True)
            self.loss_G_GAN_B = self.criterionGAN(
                pred_fake_A - pred_real_A, True)
        else:
            self.loss_G_GAN_A = (torch.mean((pred_real_A - torch.mean(pred_fake_A) + 1.0)
                                 ** 2) + torch.mean((pred_fake_A - torch.mean(pred_real_A) - 1.0) ** 2))/2
            self.loss_G_GAN_B = (torch.mean((pred_real_B - torch.mean(pred_fake_B) + 1.0)
                                 ** 2) + torch.mean((pred_fake_B - torch.mean(pred_real_B) - 1.0) ** 2))/2
        
        self.loss_G_GAN = self.loss_G_GAN_A + self.loss_G_GAN_B
            
        # L1 loss for paired data
        if mode == 'aligned':
            self.loss_G_L1_B = self.criterionL1(
                self.fake_B, self.real_B) * lambda_L1
            self.loss_G_L1_A = self.criterionL1(
                self.fake_A, self.real_A) * lambda_L1

            self.loss_G_L1 = self.loss_G_L1_A + self.loss_G_L1_B

            # self.l1_counter.append(self.loss_G_L1_B.item())  # might not be necessary
            
        # Full loss
        self.loss_G = self.loss_G_GAN + self.loss_cycle + self.loss_G_idt
        if mode == 'aligned':
            self.loss_G += self.loss_G_L1
            
        self.loss_G.backward()

    def optimize_parameters(self, mode='aligned'):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(mode)  # compute fake images and reconstruction images.
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A(mode)      # calculate gradients for D_A
        self.backward_D_B(mode)      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(mode)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
