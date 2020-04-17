from __future__ import division
import PIL.Image as pil
from archive.sig_model import *


def read_test_files(file_path, dataset_dir):
    """
    Read test files for depth estimation
    :param file_path: absolute path to test_files_*.txt
    :param dataset_dir: root path of kitti dataset
    :return: read test files with kitti root path added as prefix
    """
    with open(file_path, 'r') as f:
        test_files = f.readlines()
        test_files = [os.path.join(dataset_dir, t) for t in test_files]

    return test_files


def test_depth(opt, test_files, batch_size=1, output_dir=None):
    """
    Evaluates the given checkpoint model on depth estimation
    :param opt: arguments
    :param test_files: list of test files
    :param batch_size: test batchsize
    :param output_dir: directory to save generated dense depth
    :return: all predicted depths
    """
    input_uint8 = tf.placeholder(
        tf.uint8, [batch_size, opt.img_height, opt.img_width, 3], name='raw_input'
    )

    # RUN MODEL NETWORK
    model = SIGNetModel(opt, input_uint8, None, None)
    fetches = {"depth": model.pred_depth[0]}

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []

        for t in range(0, len(test_files), batch_size):
            the_feed_dict = {}
            inputs = np.zeros((batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8)
            for b in range(batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                # adapt to py3 ref: https://github.com/python-pillow/Pillow/issues/1605
                fh = open(test_files[idx], 'rb')
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)

                inputs[b] = np.array(scaled_im)
                inputs[b] = inputs[b] * opt.lighting_factor

            the_feed_dict[input_uint8] = inputs
            pred = sess.run(fetches, feed_dict=the_feed_dict)
            
            for b in range(batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            np.save(os.path.join(output_dir, 'model.npy'), pred_all)

        return pred_all
