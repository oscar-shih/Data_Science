import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')



## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')

parser.add_argument('--src_data_path', type = str, default = './digit/train', help = 'The directory where the input data is stored.')
parser.add_argument('--src_label_path', type = str, default = './digit/train.csv', help = 'The label file.')
parser.add_argument('--output_file', type = str, default = 'experiment/train/output.csv', help = 'The directory where the summaries will be stored.') # 'experiments/'

parser.add_argument('--job_dir', type = str, default = 'experiment/train/', help = 'The directory where the summaries will be stored.') # 'experiments/'

parser.add_argument('--pretrained', action = 'store_true', default = False, help = 'Load pretrained model')
parser.add_argument('--inference_only', action = 'store_true', default = False, help = 'Load pretrained model')

parser.add_argument('--source_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = 'model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') 


## Training
parser.add_argument('--arch', type = str, default = 'cnn', help = 'Architecture of teacher and student')  # 
parser.add_argument('--model', type = str, default = 'CNN', help = 'The target model.') # 
parser.add_argument('--wandb', type = bool, default = True) 
parser.add_argument('--num_epochs', type = int, default = 15, help = 'The num of epochs to train.') 


parser.add_argument('--train_batch_size', type = int, default = 64, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 32, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.01) #
parser.add_argument('--lr_gamma', type = float, default = 0.1)
parser.add_argument('--lr_decay_step', type = int, default = 20)

parser.add_argument('--weight_decay', type = float, default = 1e-3, help = 'The weight decay of loss.')


## Status
parser.add_argument('--print_freq', type = int, default = 500, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

