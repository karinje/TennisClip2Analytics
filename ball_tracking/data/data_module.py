import argparse
from pathlib import Path 
from fastbook import * 
import pandas as pd
import numpy as np
import logging
import re
import math

TRAIN_DATA ='/home/ubuntu/test-storage/Users/sanjaykarinje/Downloads/Dataset/'
INFER_DATA = '/home/ubuntu/test-storage/Users/sanjaykarinje//Downloads/match_frames'
NUM_INP_IMGS = 3 
OUTPUT_CLASSES = 256
IMG_RESIZE = (360,640)

logging.basicConfig(level=logging.INFO)

def all_offsets(num_imgs: int, target_pos: int):
    """returns an array of numbers where target_pos is 0 and elements to
    right are incr by 1 sequentially and elements on left and decr by 1
    """
    return np.arange(num_imgs)-target_pos+1

def offset_fname(f: Path, offset: int, prefix:str):
    """returns filename that's offset by number if it exists, else returns same file"""
    img_num = int(f.stem.replace(prefix,''))+offset
    img_f = f.parent/(prefix+str(img_num).zfill(4)+f.suffix)
    logging.debug(f'inside offset fname: {img_f} and {img_f.exists()}')
    return img_f if img_f.exists() else f

def find_prefix(f: str):
    """returns the string prefix before digits in the provided filename"""
    pattern = re.compile(r'^([a-zA-Z_]+)\d+')
    match = pattern.match(str(f))
    prefix = match.group(1) if match else ''
    logging.debug(f'filename: {f} and prefix: {prefix}')
    return prefix  
    

class BaseDataModule(object):
    """Base class for generating inputs and targets
       Inputs are file locations for input images
       Target is ball location expressed as [x,y] coordinate  
    """
    def __init__(self, args: argparse.Namespace=None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        logging.debug(f'printing args: {self.args}')
        self.train_data_path = Path(self.args.get("train_data_path", TRAIN_DATA))
        self.infer_data_path = Path(self.args.get("infer_data_path", INFER_DATA))
        self.num_inp_images = self.args.get("num_inp_images", NUM_INP_IMGS)
        self.target_pos = self.args.get("target_img_position", self.num_inp_images)
        self.target_pos = self.num_inp_images if self.target_pos is None or self.target_pos not in np.arange(1,self.num_inp_images) else self.target_pos
        self.output_classes = self.args.get("output_classes", OUTPUT_CLASSES)
        self.samples_per_batch = self.args.get("samples_per_batch", 2)
        self.validation_type = self.args.get("validation_type", "random")
        self.valid_pct = self.args.get("valid_pct", 0.2)
        self.val_games = self.args.get("val_games",('game5'))
        self.img_resize = self.args.get("img_resize", IMG_RESIZE)
        self.augment = self.args.get("augment", False)
        self.num_workers = self.args.get("num_workers", 0)

    @staticmethod
    def add_to_argparser(parser):
        parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA, help="Input path to training image files")
        parser.add_argument("--infer_data_path", type=str, default=INFER_DATA, help="Input path to inference image files")
        parser.add_argument("--num_inp_images", type=int, default=NUM_INP_IMGS, help="Number of input images per sample")
        parser.add_argument("--target_img_position", type=int, required=False, help="Position of image with target")
        parser.add_argument("--output_classes", type=int, default=OUTPUT_CLASSES, help="Number of classes in output")
        parser.add_argument("--samples_per_batch", type=int, default=2, help="Samples per batch")
        parser.add_argument("--validation_type", type=str, default='random', help="Method for selecting validation")
        parser.add_argument("--valid_pct", type=float, default=0.2, help="Select validation percent for random method")
        parser.add_argument("--val_games", type=tuple, default=('game5'), help="Folders for validation if not random")
        parser.add_argument("--img_resize", type=tuple, default=IMG_RESIZE, help="Image resize dimensions")
        parser.add_argument("--augment", type=bool, default=False, help="To turn on data augmentation")
        return parser

    def get_valid_files(self):
        def _get_valid_files(path: Path):
            if path.is_file():
                with open(str(path), 'r') as file:
                    files = [Path(line.rstrip('\n')) for line in file.readlines()]
                    files = L(files)
                    #print(f'printfing files: {files}')
            else:
                files = get_image_files(path)
            offsets = all_offsets(self.num_inp_images, self.target_pos)
            prefix = find_prefix(files[0].stem)
            logging.debug(set([offset_fname(files[0], offset, prefix=prefix) for offset in offsets]), self.num_inp_images)
            return L([f for f in files if len(set([offset_fname(f, offset, prefix=prefix) for offset in offsets]))==self.num_inp_images]) 
        return _get_valid_files

    @staticmethod
    def get_x(offset):
        def _get_x(f): 
            prefix = find_prefix(f.stem)
            return offset_fname(f, offset=offset, prefix=prefix)
        return _get_x 

    def get_xs(self):
        offsets = all_offsets(self.num_inp_images, self.target_pos)
        return [self.get_x(offset) for offset in offsets]

    def get_y(self, label_file='Label.csv'):
        def _get_y(f):
            label_file_full = f.parent/label_file
            df = pd.read_csv(label_file_full, index_col=0)
            visibility, x, y, status, shot = df.loc[f.name]
            return [x,y] if not any(map(math.isnan, (x,y))) else [0,0]
        return _get_y

    def get_splitter(self):
        return RandomSplitter(valid_pct=self.valid_pct) if self.validation_type=='random' else FuncSplitter(lambda o: o.parent.parent.name in self.val_games)

    def get_blocks(self):
        return [ImageBlock for _ in range(self.num_inp_images)]+[PointBlock]

    def get_db(self):
        batch_tfms = [*aug_transforms(max_zoom=1.05, max_rotate=5, max_warp=0)] if self.augment else []
        batch_tfms += [Normalize.from_stats(*imagenet_stats)]
        blocks, get_items, get_x, get_y, splitter, num_inp_images = self.get_blocks(), self.get_valid_files(), self.get_xs(), self.get_y(), self.get_splitter(), self.num_inp_images
        return DataBlock(blocks=blocks, get_items=get_items, get_x=get_x, get_y=get_y, splitter=splitter, n_inp=self.num_inp_images, item_tfms=[Resize(self.img_resize)], batch_tfms=batch_tfms)

    def get_dls(self, mode='training'):
        db = self.get_db()
        return db.dataloaders(self.train_data_path if mode=='training' else self.infer_data_path, bs=self.samples_per_batch, num_workers=self.num_workers)

    def config(self):
        return {"num_inp_images":self.num_inp_images, "output_classes": self.output_classes}

    def print_info(self):
        logging.info("Data Module Details-----------------------------------------------------------------------------------")
        logging.info(f'Class: {type(self).__name__}')
        logging.info(f'Input Images per Sample: {self.num_inp_images} and Target Img Position: {self.target_pos}')
        logging.info(f'# Train Imgs: {len(self.get_valid_files()(self.train_data_path))}, Source Path: {self.train_data_path}')
        logging.info(f'# Infer Imgs: {len(self.get_valid_files()(self.infer_data_path))}, Source Path: {self.infer_data_path}')
        sample_f = self.get_valid_files()(self.train_data_path)[0]
        logging.info(f'Sample File: {sample_f}')
        logging.info(f'check get_xs: {[func(sample_f) for func in self.get_xs()]}')
        logging.info(f'check get_y: {self.get_y()(sample_f)}')
        test_dls = self.get_dls()
        logging.info(f'{test_dls.valid.one_batch()[0].shape}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser = BaseDataModule.add_to_argparser(parser)
    args = parser.parse_args()
    data_mod = BaseDataModule(args)
    data_mod.print_info()
    db = data_mod.get_db()
    db.summary(data_mod.train_data_path)
    dls = data_mod.get_dls()
    logging.info(f'{dls.train.one_batch()}')
    #b = dls.train.one_batch()
    #logging.info(f'len batch: {len(b)}, b[0].shape: {b[0].shape}')
    # python ~/git/ball_tracking_3d/ball_tracking/data/data_module.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset 
    #                                                                 --infer_data_path /Users/sanjaykarinje/Downloads/match_frames 
    #                                                                 --num_inp_images 3 --target_img_position 1 

