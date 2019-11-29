import argparse
from segmentation import SegModel
import tensorflow as tf
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', dest='dataset_path',
                    default='dataset/',
                    help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=2, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--image_width', dest='image_width', type=int, default=288, help='scale images to this size')
parser.add_argument('--image_height', dest='image_height', type=int, default=288, help='then crop to this size')
parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--summary_write_freq', dest='summary_write_freq', type=int, default=10,
                    help='for how many iterations write a summary')

parser.add_argument('--test_start', dest='test_start', type=int, default=0, help='epoch from which start to test')
parser.add_argument('--test_end', dest='test_end', type=int, default=100, help='epoch on which to stop test')

parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='number of output classes')
parser.add_argument('--mode', dest='mode', default='test_val', help='train, test, test_val, freeze')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default=os.path.dirname(__file__)+'/checkpoint',
                    help='models are saved here')
parser.add_argument('--summaries_dir', dest='summaries_dir', default=os.path.dirname(__file__)+'/summary',
                    help='sample are saved here')
parser.add_argument('--test_out_dir', dest='test_out_dir', default=os.path.dirname(__file__)+'/result',
                    help='test sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default=os.path.dirname(__file__)+'/test',
                    help='read test from here')
parser.add_argument('--freezed_pb_path', dest='freezed_pb_path',
                    default=os.path.dirname(__file__)+'/test.pb',
                    help='path to save .pb')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.summaries_dir):
        os.makedirs(args.summaries_dir)

    if not os.path.exists(args.test_out_dir):
        os.makedirs(args.test_out_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = SegModel(args, sess)
        model.build_model()

        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test(args.test_out_dir, 1, ['softmax'], model.blending)
        elif args.mode == 'test_val':
            model.measure_metrics(args, model.compute_IOU)
        elif args.mode == 'freeze':
            model.freeze_save_graph(sess=sess,  output_node='mask', path='./', name='test.pb')

if __name__ == '__main__':
    tf.app.run()
