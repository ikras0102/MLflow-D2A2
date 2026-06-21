import argparse


parser = argparse.ArgumentParser()


# General setting
parser.add_argument('--model_name', type=str, default='D2A2', 
                    help='choose model class name')
parser.add_argument('--model_file', type=str, default='D2A2', 
                    help='choose model file')
parser.add_argument('--scale', type=int, default=16,##4 | 8 | 16
                    help='scale factor')
parser.add_argument('--dataset', type=str, default='nyu',
                    help='dataset nyu, middlebury, lu or rgbdd')  
parser.add_argument('--dataset_dir', type=str, default='datasets/NYUv2',#Lu | Middlebury | RGBDD | NYUv2
                    help='dataset root dir')
parser.add_argument('--pretrain_path', type=str, default=None,
                    help='load pretrained model parameters')
# Model setting
parser.add_argument('--n_resblocks', type=int, default=4,
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')

# Train setting
parser.add_argument('--input_size', type=int, default=256,
                    help='input size')
parser.add_argument('--epoch', type=int, default=600,
                    help='max epoch')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--augment', type=bool, default=True,
                    help='data augment')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--step_size', type=int, default=5000000, 
                    help='learning rate step size')
parser.add_argument('--trainresult', type=str, default='./result/trainresult/',
                    help=' train result file path')
parser.add_argument('--last_epoch', type=int, default=-1,
                    help='train epoch')
parser.add_argument('--loss', type=str, default='maskloss', #l1 | our loss
                    help='choice loss')
parser.add_argument('--validate_interval', type=int, default=5,
                    help='validate and update best checkpoint every n epochs')

# Test setting 
parser.add_argument('--save', action='store_true',#save results
                    help='save depth SR or not') 
parser.add_argument('--testresult', type=str, default='./result/testresult/', 
                    help=' train result file path')
parser.add_argument('--net_path', type=str, default=None,
                    help='load pretrained model parameters') #put model parameters




#depthanything encoder
parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])

# MLflow setting
parser.add_argument('--mlflow_experiment', type=str, default='MLflow+D2A2',
                    help='MLflow experiment name')
parser.add_argument('--mlflow_run_name', type=str, default=None,
                    help='optional MLflow run name')
parser.add_argument('--cuda_devices', type=str, default='0,1',
                    help='CUDA_VISIBLE_DEVICES used by mlflow train/test entrypoints')
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--n_trials", type=int, default=10)

parser.add_argument("--log_interval", type=int, default=50)

parser.add_argument("--save_last_checkpoint", action="store_true")
parser.add_argument("--save_all_epochs", action="store_true")
parser.add_argument("--save_mlflow_model", action="store_true")
parser.add_argument("--save_best_checkpoint_artifact", action="store_true", default=True)

# DVC metadata setting
parser.add_argument("--dvc_pull", action="store_true",
                    help="run dvc pull before building dataloaders")
parser.add_argument("--dvc_required", action="store_true",
                    help="fail early when DVC is unavailable or data paths are not present")
parser.add_argument("--dvc_remote", type=str, default=None,
                    help="optional DVC remote name used by dvc pull")
parser.add_argument("--dvc_jobs", type=int, default=None,
                    help="optional parallel jobs passed to dvc pull")
parser.add_argument("--dvc_data_paths", type=str, default=None,
                    help="optional comma-separated extra DVC-tracked data paths to log to MLflow")
parser.add_argument("--skip_dvc_metadata", action="store_true",
                    help="skip logging Git/DVC data version metadata to MLflow")

args, _unknown_args = parser.parse_known_args()
# print(args)
