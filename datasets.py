# camera-ready

import torch
import torch.utils.data

import numpy as np
import cv2
import os

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

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

        # flip the img and the label with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            label_img = cv2.flip(label_img, 1)

        ########################################################################
        # randomly scale the img and the label:
        ########################################################################
        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))
        ########################################################################

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

        ########################################################################
        # select a 256x256 random crop from the img and label:
        ########################################################################
        start_x = np.random.randint(low=0, high=(new_img_w - 256))
        end_x = start_x + 256
        start_y = np.random.randint(low=0, high=(new_img_h - 256))
        end_y = start_y + 256

        img = img[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))
        ########################################################################

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

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img) # (shape: (256, 256))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

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

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
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
        # resize img without interpolation:
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetThnSeq(torch.utils.data.Dataset):
    def __init__(self, thn_data_path):
        self.img_dir = thn_data_path + "/"

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

        # normalize the img (with mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples
