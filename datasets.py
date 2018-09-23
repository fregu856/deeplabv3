import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2
import os
from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
bdd_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'rail track'           ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'building'             , 10 ,      2 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'garage'               , 12 ,        2 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'guard rail'           , 13 ,        19 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'tunnel'               , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'wall'                 , 15 ,      3 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'banner'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'billboard'            , 17 ,        19 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'lane divider'         , 18 ,      7 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'parking sign'         , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'polegroup'            , 21 ,        19 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'street light'         , 22 ,        5 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'traffic cone'         , 23 ,       7 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'traffic device'       , 24 ,       7 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'traffic light'        , 25 ,       6 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'traffic sign'         , 26 ,       7 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'traffic sign frame'   , 27 ,       7 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'terrain'              , 28 ,       9 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'vegetation'           , 29 ,      8 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'sky'                  , 30 ,      10 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'person'               , 31 ,       11 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'rider'                , 32 ,       12 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'trailer'              , 38 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# create a function which maps id to trainId:
bdd_id_to_trainId = {label.id: label.trainId for label in bdd_labels}
bdd_id_to_trainId_map_func = np.vectorize(bdd_id_to_trainId.get)

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
# train_dirs = ["aachen/"]
# val_dirs = ["lindau/"]
# test_dirs = ["bonn"]

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.num_train_classes = 20 # (road, sidewalk, car etc.)

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))

        ########################################################################
        # flip the img and the label with 0.5 probability:
        ########################################################################
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)

            label_img = cv2.flip(label_img, 1)

        ########################################################################

        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.img_h)
        new_img_w = int(scale*self.img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))

        # # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        start_x = np.random.randint(low=0, high=(new_img_w - 512))
        end_x = start_x + 512
        start_y = np.random.randint(low=0, high=(new_img_h - 512))
        end_y = start_y + 512

        img = img[start_y:end_y, start_x:end_x] # (shape: (512, 512, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (512, 512))

        # # # # # # # # debug visualization START
        # print (img.shape)
        # print (label_img.shape)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 512, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 512))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 512))
        label_img = torch.from_numpy(label_img) # (shape: (512, 512))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

# test = DatasetTrain("/home/fregu856/exjobb/data/cityscapes", "/home/fregu856/exjobb/data/cityscapes/meta")
# for i in range(10):
#     _ = test.__getitem__(i)

class DatasetTrain_small(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.num_train_classes = 20 # (road, sidewalk, car etc.)

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

        ########################################################################
        # flip the img and the label with 0.5 probability:
        ########################################################################
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)

            label_img = cv2.flip(label_img, 1)

        ########################################################################

        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))

        # # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        start_x = np.random.randint(low=0, high=(new_img_w - 256))
        end_x = start_x + 256
        start_y = np.random.randint(low=0, high=(new_img_h - 256))
        end_y = start_y + 256

        img = img[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))

        # # # # # # # # debug visualization START
        # print (img.shape)
        # print (label_img.shape)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img) # (shape: (256, 256))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

# test = DatasetTrain_small("/home/fregu856/exjobb/data/cityscapes", "/home/fregu856/exjobb/data/cityscapes/meta")
# for i in range(10):
#     _ = test.__getitem__(i)

class DatasetTrain_BDD_small(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.num_train_classes = 20 # (road, sidewalk, car etc.)

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.bdd_img_dir_train = "/datasets/bdd/segmentation/train/raw_images/"
        self.bdd_img_dir_val = "/datasets/bdd/segmentation/val/raw_images/"
        self.bdd_label_dir_train = "/datasets/bdd/segmentation/train/class_id/"
        self.bdd_label_dir_val = "/datasets/bdd/segmentation/val/class_id/"
        # self.bdd_img_dir_train = "/home/fregu856/exjobb/data/bdd/segmentation/train/raw_images/"
        # self.bdd_img_dir_val = "/home/fregu856/exjobb/data/bdd/segmentation/val/raw_images/"
        # self.bdd_label_dir_train = "/home/fregu856/exjobb/data/bdd/segmentation/train/class_id/"
        # self.bdd_label_dir_val = "/home/fregu856/exjobb/data/bdd/segmentation/val/class_id/"

        bdd_img_file_names_train = os.listdir(self.bdd_img_dir_train)
        for img_file_name in bdd_img_file_names_train:
            img_id = img_file_name.split(".jpg")[0]

            img_path = self.bdd_img_dir_train + img_file_name

            label_img_path = self.bdd_label_dir_train + img_id + ".png"

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)

        bdd_img_file_names_val = os.listdir(self.bdd_img_dir_val)
        for img_file_name in bdd_img_file_names_val:
            img_id = img_file_name.split(".jpg")[0]

            img_path = self.bdd_img_dir_val + img_file_name

            label_img_path = self.bdd_label_dir_val + img_id + ".png"

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)

        if img.shape == (1024, 2048, 3): # (cityscapes)
            # resize img without interpolation (want the image to still match
            # label_img, which we resize below):
            img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                             interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

            label_img_path = example["label_img_path"]
            label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
            # resize label_img without interpolation (want the resulting image to
            # still only contain pixel values corresponding to an object class):
            label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                                   interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

        elif img.shape == (720, 1280, 3): # (bdd)
            img = img[0:640, :, :] # (shape: (640, 1280, 3))
            # resize img without interpolation (want the image to still match
            # label_img, which we resize below):
            img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                             interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

            label_img_path = example["label_img_path"]
            label_img = cv2.imread(label_img_path, -1) # (shape: (720, 1280))
            label_img = label_img[0:640, :] # (shape: (640, 1280))
            # resize label_img without interpolation (want the resulting image to
            # still only contain pixel values corresponding to an object class):
            label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                                   interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

            label_img = bdd_id_to_trainId_map_func(label_img) # (shape: (512, 1024))
            label_img = label_img.astype(np.uint8)

        else:
            print ("Unknown image shape!")
            print (img.shape)
            return self.__getitem__(0)

        ########################################################################
        # flip the img and the label with 0.5 probability:
        ########################################################################
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)

            label_img = cv2.flip(label_img, 1)

        ########################################################################

        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))

        # # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        start_x = np.random.randint(low=0, high=(new_img_w - 256))
        end_x = start_x + 256
        start_y = np.random.randint(low=0, high=(new_img_h - 256))
        end_y = start_y + 256

        img = img[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))

        # # # # # # # # debug visualization START
        # print (img.shape)
        # print (label_img.shape)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img) # (shape: (256, 256))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

# test = DatasetTrain_BDD_small("/home/fregu856/exjobb/data/cityscapes", "/home/fregu856/exjobb/data/cityscapes/meta")
# for i in range(100):
#     _ = test.__getitem__(i)

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.num_train_classes = 20 # (road, sidewalk, car etc.)

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"
                label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

        # # # # # # # # debug visualization START
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))
        label_img = torch.from_numpy(label_img) # (shape: (512, 1024))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

# test = DatasetVal("/home/fregu856/exjobb/data/cityscapes", "/home/fregu856/exjobb/data/cityscapes/meta")
# for i in range(10):
#     _ = test.__getitem__(i)

class DatasetEvalVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.num_train_classes = 20 # (road, sidewalk, car etc.)

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"
                label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

        # # # # # # # # debug visualization START
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))
        label_img = torch.from_numpy(label_img) # (shape: (512, 1024))

        return (img, label_img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, sequence):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/demoVideo/stuttgart_" + sequence + "/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split("_leftImg8bit.png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetThnSeq(torch.utils.data.Dataset):
    def __init__(self):
        self.img_dir = "/staging/frexgus/test/"

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (512, 1024, 3))
        # img = cv2.resize(img, (2048, 1024),
        #                  interpolation=cv2.INTER_NEAREST)

        ########################################################################
        # normalize the img (with mean and std for the pretrained ResNet):
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples
