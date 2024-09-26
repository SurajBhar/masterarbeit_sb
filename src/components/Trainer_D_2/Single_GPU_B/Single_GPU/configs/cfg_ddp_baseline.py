import configargparse
import os

def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', 
                        is_config_file=True, 
                        default='/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/Trainer_D_2/Multi_GPU_B/Multi_GPU/configs/baseline_configs/base_B_01_Adam_0_03_split_0.txt',
                        help='config file path')
    
    parser.add_argument('--checkpoint_path', 
                        type=str,
                        default=None,
                        help='checkpoint_path to resume training from a specific checkpoint')

    parser.add_argument("--experiment_name", 
                        type=str, 
                        default=None,
                        help='Experiment name. If left blank, a name will be auto-generated.')
    
    parser.add_argument("--features_path", 
                        type=str, 
                        default="/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/Trainer_D_2/Dataloader_2/features_store/rgb_split_0_daa/all_split_0_1024_features.pkl", 
                        help="Path to the features data directory")

    parser.add_argument("--labels_path", 
                        type=str, 
                        default="/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/Trainer_D_2/Dataloader_2/features_store/rgb_split_0_daa/all_split_0_1024_labels.pkl", 
                        help="Path to the ground truth labels data directory")
    
    parser.add_argument("--img_path", 
                        type=str, 
                        default="/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/Trainer_D_2/Dataloader_2/features_store/rgb_split_0_daa/all_split_0_1024_imagepaths.pkl", 
                        help="Path to the training images path data directory")
    
    parser.add_argument("--train_dir", 
                        type=str, 
                        default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/train", 
                        help="Path to the training data directory")
    
    parser.add_argument("--val_dir", 
                        type=str, 
                        default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/val", 
                        help="Path to the validation data directory")

    parser.add_argument("--test_dir", 
                        type=str, 
                        default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/test", 
                        help="Path to the test data directory")
    
    # Training and optimization options
    parser.add_argument("--device", 
                        type=int, 
                        default=0, 
                        help="GPU Number for running the experiment")
    
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=100, 
                        help="Number of epochs for training")
    
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=10, 
                        help=" The number of worker threads for data loading")
    
    parser.add_argument("--prefetch_factor", 
                        type=int, 
                        default=None, 
                        help="Each worker pre-fetches 2 batches of data in advance")

    parser.add_argument("--num_classes", 
                        type=int, 
                        default=34, 
                        help="Number of epochs for training")
    
    parser.add_argument("--save_every", type=int, default=10, help='How often to save a snapshot')

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1024, 
                        help="Batch size for training")
    
    parser.add_argument("--optimizer", 
                        choices=["ADAM", "SGD"], 
                        default="SGD", 
                        help="Choose optimizer (adam or sgd)")
    
    parser.add_argument('--scheduler','-sc',
                        type=str,
                        default='CosineAnnealingLR',
                        choices=['LambdaLR','CosineAnnealingLR'],
                        help='Switch between LambdaLR and CosineAnnealingLR')

    parser.add_argument("--resume", 
                        action="store_true", 
                        default=False, 
                        help="Resume training from checkpoint")

    parser.add_argument('--lr', 
                        type=float, 
                        default=0.03,
                        help='Initial learning rate')
    
    parser.add_argument('--momentum', 
                        type=float, 
                        default=0.9,
                        help='Momentum')
    
    parser.add_argument('--weight_decay', '-wd', 
                        type=float, 
                        default=0.0,
                        help='Weight decay (L2 penalty)')
    
    parser.add_argument('--w_decay_adam', '-wda', 
                        type=float, 
                        default=0.0,
                        help='Weight decay Adam optimizer (L2 penalty)')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    # Auto-generate experiment name if not provided
    if cfg.experiment_name is None:
        cfg.experiment_name = '{}-{}-{}-{}'.format(os.path.basename(cfg.train_dir), 
                                            cfg.optimizer,
                                            cfg.scheduler,
                                            cfg.lr)
    return cfg
