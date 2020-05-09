
import numpy as np
import tensorflow as tf
import sys
from pba.sig_nets import *
from pba.sig_utils import *


class Model(object):
    """Builds an model."""

    def __init__(self, hparams, mode="train"):
        self.hparams = hparams
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        assert mode in ['train', 'eval', 'test']
        self.is_training = True if mode == "train" else False
        self.mode = mode

    def build(self):
        """Construct the model."""
        self._setup_misc(self.mode)
        self._setup_images_and_intrinsic(self.mode)
        self._build_graph(self.mode)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def _setup_misc(self, mode):
        """Sets up miscellaneous in the model constructor."""
        self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
        if mode == "train":
            self.batch_size = self.hparams.batch_size
        self.test_batch_size = self.hparams.test_batch_size

    def _setup_images_and_intrinsic(self, mode):
        """Sets up input placeholders for the model."""
        ns = self.hparams.num_source
        if mode == 'train':
            # variables used for photo loss on original images
            self.tgt_image_input = tf.placeholder(
                tf.float32, [self.batch_size, self.input_height, self.input_width, 3]
            )
            self.src_image_stack_input = tf.placeholder(
                tf.float32, [self.batch_size, self.input_height, self.input_width, 3 * ns]  # assuming two source images
            )
            self.intrinsic_input = tf.placeholder(tf.float32, [self.batch_size, 3, 3])
        else:
            self.tgt_image_input = tf.placeholder(
                tf.float32, [None, self.input_height, self.input_width, 3]
            )

    def _build_graph(self, mode):
        """Constructs the TF graph for the model.

        Args:
          mode: string indicating training mode ( e.g., 'train', 'eval').
        """
        if self.is_training:
            self.global_step = tf.train.get_or_create_global_step()

        # Build train/eval model
        self.build_model()
        self._calc_num_trainable_params()

        if self.is_training:
            self._build_train_op()

        # Setup checkpointing for this child model
        # Keep 2 or more checkpoints around during training.
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(max_to_keep=10)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def signet_data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:, 0, 0] * x_scaling
            fy = intrinsics[:, 1, 1] * y_scaling
            cx = intrinsics[:, 0, 2] * x_scaling
            cy = intrinsics[:, 1, 2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics, [out_h, out_w]

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics, [offset_y, offset_x, out_h, out_w]

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            im_aug = im_f ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            im_aug = im_aug * random_brightness

            # TODO: apply same color shift for all frames
            # randomly shift color
            in_c = 3
            random_colors = tf.random_uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)

            color_image = tf.concat([color_image, color_image, color_image], axis=3)
            im_aug *= color_image

            # saturate
            im_aug = tf.clip_by_value(im_aug, 0, 1)
            im_aug = tf.image.convert_image_dtype(im_aug, tf.float32)

            return im_aug

        im, intrinsics, out_hw = random_scaling(im, intrinsics)
        im, intrinsics, yxhw = random_cropping(im, intrinsics, out_h, out_w)
        do_augment = tf.random_uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)

        return im, intrinsics

    def build_model(self):
        opt = self.hparams

        if self.mode == "train":
            tf.logging.info("Building train model")
            image_all = tf.concat([self.tgt_image_input, self.src_image_stack_input], axis=3)
            # kitti augmentation is enabled
            if opt.use_kitti_aug:
                image_all, self.intrinsic_input = self.signet_data_augmentation(
                    image_all, self.intrinsic_input, opt.input_height, opt.input_width
                )
            # image_channels=3*opt.num_source
            self.tgt_image = self.preprocess_image(image_all[:, :, :, :3])
            self.src_image_stack = self.preprocess_image(image_all[:, :, :, 3:])

            self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
            self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) for img in self.tgt_image_pyramid]

            self.src_image_concat = tf.concat(
                [self.src_image_stack[:, :, :, 3 * i:3 * (i + 1)] for i in range(opt.num_source)], axis=0
            )
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)
            self.intrinsics = self.get_multi_scale_intrinsics(
                self.intrinsic_input, self.hparams.num_scales
            )
            self.pred_depth = self.build_dispnet()
            self.pred_poses  = self.build_posenet()
            self.build_rigid_flow_warping()
            self.build_losses()
        else:
            self.tgt_image = self.preprocess_image(self.tgt_image_input)
            # building model in eval mode
            tf.logging.info("Building eval model")
            self.pred_depth = self.build_dispnet()

    def _calc_num_trainable_params(self):
        # TODO check for correctness for eval mode case
        self.num_trainable_params = np.sum(
            [np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()]
        )
        tf.logging.info('number of trainable params: {}'.format(self.num_trainable_params))

    def _build_train_op(self):
        """Builds the train op for the model."""
        hparams = self.hparams
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.total_loss, tvars)
        if hparams.gradient_clipping_by_global_norm > 0.0:
            grads, norm = tf.clip_by_global_norm(grads, hparams.gradient_clipping_by_global_norm)
            # tf.summary.scalar('grad_norm', norm)

        # Setup the initial learning rate
        if self.hparams.optimizer == "sgd":
            tf.logging.info("Using SGD optimizer! Set lr_decay=cosine for best results!")
            self.optimizer = tf.train.MomentumOptimizer(self.lr_rate_ph, 0.9, use_nesterov=True)
        else:
            tf.logging.info("Defaulting to Adam optimizer")
            self.optimizer = tf.train.AdamOptimizer(self.lr_rate_ph, 0.9)

        apply_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
        train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies([apply_op]):
            self.train_op = tf.group(*train_ops)

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        """
        :param fx: focal length x
        :param fy: focal length y
        :param cx: image center coordinate x
        :param cy: image center coordinate y
        :return: scaled intrinsics
        """
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)

        return intrinsics

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0] / (2 ** s)
            fy = intrinsics[:, 1, 1] / (2 ** s)
            cx = intrinsics[:, 0, 2] / (2 ** s)
            cy = intrinsics[:, 1, 2] / (2 ** s)
            intrinsics_mscale.append(self.make_intrinsics_matrix(fx, fy, cx, cy))

        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    def preprocess_image(self, image):
        """
        Converts image to range [-1, 1]
        :param image: image to be converted can be float(0, 1) or int(0, 255)
        :return: image converted in range [-1, 1]
        """
        # Assuming input image is uint8
        assert image is not None
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.

    def build_dispnet(self):
        opt = self.hparams

        # build dispnet_inputs
        if self.mode == "train":
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat(
                    [self.dispnet_inputs, self.src_image_stack[:, :, :, 3 * i:3 * (i + 1)]], axis=0
                )
        else:
            # for eval/test depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image

        self.pred_disp = disp_net(opt, self.dispnet_inputs, self.is_training)
        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        pred_depth = [1. / d for d in self.pred_disp]

        # TODO Add multi-scale depth maps to TF summary.
        # for i in range(len(pred_depth)):
        #    tf.summary.image('pred_depth_' + str(i), pred_depth[i], max_outputs=opt.max_outputs)

        return pred_depth

    def build_posenet(self):
        opt = self.hparams
        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        pred_poses = pose_net(opt, self.posenet_inputs, self.is_training)
        return pred_poses

    def build_rigid_flow_warping(self):
        opt = self.hparams
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
                    tf.squeeze(self.pred_depth[s][bs * (i + 1):bs * (i + 2)], axis=3),
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

        """
        # TODO Record forward rigid flow warping result on tensorboard
        for i in range(len(self.fwd_rigid_warp_pyramid)):
            tf.summary.image(
                "fwd_rigid_warp_scale" + str(i), self.fwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs
            )
        # TODO Record backward rigid flow warping result on tensorboard
        for i in range(len(self.bwd_rigid_warp_pyramid)):
            tf.summary.image(
                "bwd_rigid_warp_scale" + str(i), self.bwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs
            )
        """

        # compute reconstruction errors based on the rigid flow
        self.fwd_rigid_error_pyramid = [
            self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) for s in
            range(opt.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]) for s in
            range(opt.num_scales)
        ]

        # TODO Record fwd rigid flow warp error on tensorboard
        self.fwd_rigid_error_scale = []
        self.bwd_rigid_error_scale = []
        for i in range(len(self.fwd_rigid_error_pyramid)):
            tmp_fwd_rigid_error_scale = tf.reduce_mean(self.fwd_rigid_error_pyramid[i], axis=3, keepdims=True)
            # tf.summary.image("fwd_rigid_error_scale" + str(i), tmp_fwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.fwd_rigid_error_scale.append(tmp_fwd_rigid_error_scale)
        # TODO Record bwd rigid flow warp error on tensorboard
        for i in range(len(self.bwd_rigid_error_pyramid)):
            tmp_bwd_rigid_error_scale = tf.reduce_mean(self.bwd_rigid_error_pyramid[i], axis=3, keepdims=True)
            # tf.summary.image("bwd_rigid_error_scale" + str(i), tmp_bwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.bwd_rigid_error_scale.append(tmp_bwd_rigid_error_scale)

    def build_losses(self):
        opt = self.hparams
        rigid_warp_loss = 0.0
        disp_smooth_loss = 0.0

        for s in range(opt.num_scales):
            # rigid_warp_loss
            if opt.rigid_warp_weight > 0:
                rigid_warp_loss += opt.rigid_warp_weight * opt.num_source / 2 * \
                                   (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) +
                                    tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))
            # disp_smooth_loss
            if opt.disp_smooth_weight > 0:
                disp_smooth_loss += opt.disp_smooth_weight / (2 ** s) * self.compute_smooth_loss(
                    self.pred_disp[s], tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0)
                )
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0.0
        if opt.use_regularization:
            self.total_loss += regularization_loss

        self.img_loss = 0.0
        self.rigid_warp_loss = 0.0
        self.disp_smooth_loss = 0.0

        self.rigid_warp_loss += rigid_warp_loss
        self.disp_smooth_loss += disp_smooth_loss
        self.img_loss = rigid_warp_loss + disp_smooth_loss
        self.total_loss += self.img_loss

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y, mask=None):
        # TODO here our mask has one channel, and img has 3 channels
        if mask != None:
            x = x * tf.tile(mask, [1, 1, 1, 3])
            y = y * tf.tile(mask, [1, 1, 1, 3])
        return self.hparams.alpha_recon_image * self.SSIM(x, y) + (1 - self.hparams.alpha_recon_image) * tf.abs(x - y)

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1, 2, 3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp / disp_mean

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
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
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
