import configargparse
import os

def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', 
                        is_config_file=True, 
                        default='/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/config_manager/experiment_1.txt',
                        help='config file path')
    
    parser.add_argument('--checkpoint_path', 
                        type=str,
                        default=None,
                        help='checkpoint_path to resume training from a specific checkpoint')

    parser.add_argument("--experiment_name", 
                        type=str, 
                        default=None,
                        help='Experiment name. If left blank, a name will be auto-generated.')

    parser.add_argument("--kir_test_dir", 
                        type=str, 
                        default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/test", 
                        help="Path to the test data directory")

    parser.add_argument("--nir_test_dir", 
                        type=str, 
                        default="/net/polaris/storage/deeplearning/sur_data/rgb_daa/split_0/test", 
                        help="Path to the test data directory")
    
    # Training and optimization options
    parser.add_argument("--device", 
                        type=int, 
                        default=0, 
                        help="GPU Number for running the experiment")

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
                        default=2, 
                        help="Number of classes based on the task. 2 for binary classification.")

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1024, 
                        help="Batch size for training")

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
