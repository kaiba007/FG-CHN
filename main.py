# Time: 2022-12-20-11-22
# Author: Xianxian Zeng
# Name: main.py
# Details: Model with Pytorch-Lightning for Fine-graiend Hashing

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import argparse

from datacsv import DInterface
from model import HInterface

def main(config):
    data_module = DInterface(config=config)
    model = HInterface(config=config)

    # training
    saving_dir = "log/%s_%s/%s_%s" % (config.model_name, config.dataset, config.code_length, config.dataset)
    checkpoint_callback = ModelCheckpoint(dirpath=saving_dir,
                                          filename='{epoch}-{val_mAP:.4f}',
                                          save_top_k=1,
                                          monitor='val_mAP',
                                          mode='max')
    csv_dir="log/%s_%s" % (config.model_name, config.dataset)
    csv_name = "%s_%s" % (config.code_length, config.dataset)
    logger = CSVLogger(save_dir=csv_dir,name=csv_name)
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=config.gpu, 
                         precision=16, 
                        #  limit_train_batches=0.5
                         default_root_dir=saving_dir,
                         num_sanity_val_steps=-1,
                         max_epochs=300,
                         callbacks=checkpoint_callback,
                         logger=logger,
                        )
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset selecting
    parser.add_argument('--dataset', default='cub', help='select dataset for experiment')
    # lr
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training')
    # batch-size
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    # epoch
    parser.add_argument('--epoch', default=300, type=int, help='epoch for training')
    # gpu
    parser.add_argument('--gpu', default=[0], help='gpu-id')
    # num_workers
    parser.add_argument('--num_workers', default=8, type=int, help='number workers for training')
    # model
    parser.add_argument('--model_name', default='resnet50', type=str, help='backbone model for experiment')
    # code length
    parser.add_argument('--code_length', default=32, type=int, help='code length for experiment')

    args = parser.parse_args()
    config = args
    if config.dataset == 'cub':
        config.classlen = 200
        config.data_root = '/workspace/data/CUB_200_2011'
        config.train_csv = './datacsv/cub/train.csv'
        config.test_csv = './datacsv/cub/test.csv'
    elif config.dataset == 'aircraft':
        config.classlen = 100
        config.data_root = '/workspace/data/FGVC/data'
        config.train_csv = './datacsv/aircraft/train.csv'
        config.test_csv = './datacsv/aircraft/test.csv'
    elif config.dataset == 'food101':
        config.classlen = 101
        config.data_root = '/workspace/dataset/fine-grained-dataset/food-101/images'
        config.train_csv = './datacsv/food101/train.csv'
        config.test_csv = './datacsv/food101/test.csv'
    elif config.dataset == 'nabirds':
        config.classlen = 555
        config.data_root = '/workspace/dataset/fine-grained-dataset/nabirds/'
        config.train_csv = './datacsv/nabirds/train.csv'
        config.test_csv = './datacsv/nabirds/test.csv'
    elif config.dataset == 'vegfru':
        config.classlen = 292
        config.data_root = '/workspace/dataset/fine-grained-dataset/vegfru-dataset/'
        config.train_csv = './datacsv/vegfru/train.csv'
        config.test_csv = './datacsv/vegfru/test.csv'
    else:
        print("We have not provided the experiments of %s" % config.dataset)

    main(config)

    

