from __future__ import division
from archive.sig_nets import *
from archive.utils import *
# TODO randomness
import tensorflow as tf
import random

class SIGNetModel(object):

    def __init__(
            self, opt, tgt_image, src_image_stack, intrinsics, add_dispnet=True, add_posenet=False, mode="train"
    ):
        self.opt = opt
        self.is_training = True if mode == "train" else False
        self.mode = mode
        self.add_dispnet = add_dispnet
        self.add_posenet = add_posenet
        #TODO random
        seed = 8964
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.intrinsics = intrinsics

        self.build_model()

        if not self.is_training:
            return

        self.build_losses()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) for img in self.tgt_image_pyramid]

        # src images concated along batch dimension
        if self.src_image_stack is not None:
            self.src_image_concat = tf.concat(
                [self.src_image_stack[:, :, :, 3*i:3*(i+1)] for i in range(opt.num_source)], axis=0
            )
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        if self.add_dispnet:
            self.build_dispnet()

        if self.add_posenet:
            self.build_posenet()

        if self.add_dispnet and self.add_posenet:
            self.build_rigid_flow_warping()

    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if self.mode == 'test':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:, :, :, 3*i:3*(i+1)]], axis=0)

        self.pred_disp = disp_net(opt, self.dispnet_inputs, False)
        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can 
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1./d for d in self.pred_disp]
        
        #TODO Add multi-scale depth maps to TF summary.
        for i in range(len(self.pred_depth)):
            tf.summary.image('pred_depth_' + str(i), self.pred_depth[i], max_outputs=opt.max_outputs)

    def build_posenet(self):
        opt = self.opt
        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        self.pred_poses = pose_net(opt, self.posenet_inputs, False)

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        for s in range(opt.num_scales):
            for i in range(opt.num_source):
                fwd_rigid_flow = compute_rigid_flow(
                        tf.squeeze(self.pred_depth[s][:bs], axis=3),
                        self.pred_poses[:, i, :], self.intrinsics[:, s, :, :], False
                )
                bwd_rigid_flow = compute_rigid_flow(
                        tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                        self.pred_poses[:, i, :], self.intrinsics[:, s, :, :], True
                )
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)

        # warping by rigid flow
        self.fwd_rigid_warp_pyramid = [
            flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s]) for s in range(opt.num_scales)
        ]
        self.bwd_rigid_warp_pyramid = [
            flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s]) for s in range(opt.num_scales)
        ]

        #TODO Record forward rigid flow warping result on tensorboard
        for i in range(len(self.fwd_rigid_warp_pyramid)):
            tf.summary.image("fwd_rigid_warp_scale" + str(i), self.fwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs)
        #TODO Record backward rigid flow warping result on tensorboard
        for i in range(len(self.bwd_rigid_warp_pyramid)):
            tf.summary.image("bwd_rigid_warp_scale" + str(i), self.bwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs)

        # compute reconstruction error  
        self.fwd_rigid_error_pyramid = [
            self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) for s in range(opt.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]) for s in range(opt.num_scales)
        ]

        #TODO Record fwd rigid flow warp error on tensorboard
        self.fwd_rigid_error_scale = []
        self.bwd_rigid_error_scale = []
        for i in range(len(self.fwd_rigid_error_pyramid)):
            tmp_fwd_rigid_error_scale = tf.reduce_mean(self.fwd_rigid_error_pyramid[i], axis=3, keepdims=True)
            tf.summary.image("fwd_rigid_error_scale" + str(i), tmp_fwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.fwd_rigid_error_scale.append(tmp_fwd_rigid_error_scale)
        #TODO Record bwd rigid flow warp error on tensorboard
        for i in range(len(self.bwd_rigid_error_pyramid)):
            tmp_bwd_rigid_error_scale = tf.reduce_mean(self.bwd_rigid_error_pyramid[i], axis=3, keepdims=True)
            tf.summary.image("bwd_rigid_error_scale" + str(i), tmp_bwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.bwd_rigid_error_scale.append(tmp_bwd_rigid_error_scale)

    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        rigid_warp_loss = 0.0
        disp_smooth_loss = 0.0

        for s in range(opt.num_scales):
            # rigid_warp_loss
            if opt.mode == 'train_rigid' and opt.rigid_warp_weight > 0:
                rigid_warp_loss += opt.rigid_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) +
                                 tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))

            # disp_smooth_loss
            if opt.mode == 'train_rigid' and opt.disp_smooth_weight > 0:
                disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0.0
        if opt.use_regularization:
            self.total_loss += regularization_loss
        self.img_loss = 0.0
        self.rigid_warp_loss = 0.0
        self.disp_smooth_loss = 0.0

        #TODO modified loss function
        if opt.mode == 'train_rigid':
            self.rigid_warp_loss += rigid_warp_loss
            self.disp_smooth_loss += disp_smooth_loss
            self.img_loss = rigid_warp_loss + disp_smooth_loss
            self.total_loss += self.img_loss

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y, mask=None):
        #TODO here our mask has one channel, and img has 3 channels
        if mask!=None:
            x=x*tf.tile(mask, [1, 1, 1, 3])
            y=y*tf.tile(mask, [1, 1, 1, 3])
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def preprocess_image(self, image):
        """
        Converts image to range [-1, 1]
        :param image: image in unit8 data format
        :return: image converted in range [-1, 1]
        """
        # Assuming input image is uint8
        if image is None:
            return None

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.
            
    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)