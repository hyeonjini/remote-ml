import argparse
import os
import trainer
import inference

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Data and Model checkpoints directories
    parser.add_argument('--model', type=str, default='BaseCNN', help='model type (default: VGG-11)')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset type (default: MNISTDataset)')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epoch (default: 10)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='dataset augmnetation type(default: BaseAugmentation')
    parser.add_argument('--resize', type=int, nargs="+", default=[32, 32], help='data augmentation resize size (default: 32,32)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validation (default: 0.2)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=1, help='num worker for dataloader (default: 4)')
    parser.add_argument('--name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    #parser.add_argument('', type=None, default=0, help='')

    # Hyperparameter environment
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--criterion', type=str, default='CrossEntropy', help='criterion type (default: CrossEntropy)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler decay step (default: 5)')

    # Container envirionment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/code/web_projects/ui_ml/ml/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/code/web_projects/ui_ml/ml/models'))

    # Custom Environment
    parser.add_argument('--early_stop', default=True, action='store_false', help='early stopping, if it no more get a better from training (default: True)')
    parser.add_argument('--task', type=str, default='Classification', help='task tpye (option: Classification, Segmentation, Generative (default: Classification))')

    # Inference enviroment
    parser.add_argument('--sample', type=int, default=3, help='count of sample image')
    parser.add_argument('--output_size', type=int, nargs="+", default=[224,224], help='visualization output size (default: 32,32)')

    args = parser.parse_args()
    
    log = None
    # start learning point
    if args.task == 'Classification':
        log = trainer.classification_training(args)
    elif args.task == 'Segmentation':
        log = trainer.segmentation_training()
    else:
        log = trainer.generative_model_training()

    # inference point
    #inference.generate_grad_cam(args, log)
    

