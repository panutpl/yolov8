from PIL import Image,ImageDraw,ImageFont
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import yaml


class loadDataset(Dataset):
    def __init__(self,dataframe,img_path:str,x_col:str,y_col:str,box_col:list,subset=None,split=None):
        self.dataframe = dataframe
        self.img_path = img_path
        self.x_col = x_col
        self.y_col = y_col

#         if box_col == ['x_min','y_min','x_max','y_max']:
#             self.box_col = box_col
#         else:print("!Please use format box_col :['x_min','y_min','x_max','y_max']!")
        self.box_col = box_col   
            

        self.subset = subset

        le = preprocessing.LabelEncoder()
        le = list(le.fit(self.dataframe[self.y_col]).classes_)
        self.label_ = {le[i]: i for i in range(len(le))}
        self.colors = ['#%02x%02x%02x' % tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(le))]
        self.dataframe_gb = self.dataframe.groupby([self.img_path,self.x_col]).size().reset_index(name='counts')


        ### set ymal
        os.makedirs('data/', exist_ok=True)
        current_path = os.getcwd()
        data_yaml = dict(train = current_path+"/datasets/train",
                        val = current_path+"/datasets/valid",
                        nc = len(le),
                        names = le
                        )
        with open('data/data.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)

        ### testset
        data_yaml = dict(train = current_path+"/datasets/train",
                        val = current_path+"/datasets/test",
                        nc = len(le),
                        names = le
                        )
        with open('data/data_test.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)
        ##########


        if subset is not None and split is not None:
            ### spilt (train,test,val)
            datafame_groupby =  self.dataframe.groupby([self.img_path,self.x_col]).size().reset_index(name='counts')
            if len(split) == 2:
                df_train, df_test = train_test_split(datafame_groupby,test_size=split[1], random_state=42)
                if subset == "train":
                    self.dataframe_gb = df_train.reset_index(drop=True)
                    os.makedirs('datasets/train/images', exist_ok=True)
                    os.makedirs('datasets/train/labels', exist_ok=True)
                elif subset == "valid": 
                    self.dataframe_gb = df_test.reset_index(drop=True)
                    os.makedirs('datasets/valid/images', exist_ok=True)
                    os.makedirs('datasets/valid/labels', exist_ok=True)
            elif len(split) == 3:
                df_train, df_test = train_test_split(datafame_groupby,test_size=split[1], random_state=42)
                df_train, df_val = train_test_split(df_train, test_size=split[2], random_state=42)
                if subset == "train":
                    self.dataframe_gb = df_train.reset_index(drop=True)
                    os.makedirs('datasets/train/images', exist_ok=True)
                    os.makedirs('datasets/train/labels', exist_ok=True)
                elif subset == "test": 
                    self.dataframe_gb = df_test.reset_index(drop=True)
                    os.makedirs('datasets/test/images', exist_ok=True)
                    os.makedirs('datasets/test/labels', exist_ok=True)
                elif subset == "valid": 
                    self.dataframe_gb = df_val.reset_index(drop=True)
                    os.makedirs('datasets/valid/images', exist_ok=True)
                    os.makedirs('datasets/valid/labels', exist_ok=True)

    def __len__(self):
        return len(self.dataframe_gb)

    def __getitem__(self,index):
      label_ = list(self.label_.keys())
      path_img = self.dataframe_gb[self.img_path][index]
      img_id = self.dataframe_gb[self.x_col][index]
      img = Image.open(path_img+img_id).convert("RGB")
      return img

    def get_dataframe(self):
      if self.subset==None:
        return self.dataframe
      return self.dataframe_gb
    

    def dataloader(self):
        copyimage_path = f'datasets/{self.subset}/images/'
        labels_path = f'datasets/{self.subset}/labels/'
        for index in tqdm(range(len(self.dataframe_gb))):
            path_img = self.dataframe_gb[self.img_path][index]
            img_id = self.dataframe_gb[self.x_col][index]
            img = Image.open(path_img+img_id).convert("RGB")
            w,h = img.size
            records = self.dataframe[self.dataframe[self.x_col] == img_id]
            boxes = records[self.box_col].values
            labels = records[self.y_col].values
            label_name = img_id[:-4]+'.txt'
            list_file = open(labels_path +label_name, 'w')
            for j in range(len(boxes)):
                class_ = self.label_[labels[j]] ### cat >> 0
                x1, y1 = boxes[j][0]/w, boxes[j][1]/h
                x2, y2 = boxes[j][2]/w ,boxes[j][3]/h
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                list_file.write(
                            f"{class_} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
                          )
                copyfile(path_img+img_id,copyimage_path+img_id)

    def augmentation_HVFlip(self,HorizontalFlip=False,VerticalFlip=False):
        copyimage_path = f'datasets/{self.subset}/images/'
        labels_path = f'datasets/{self.subset}/labels/'
        for index in tqdm(range(len(self.dataframe_gb))):
            path_img = self.dataframe_gb[self.img_path][index]
            img_id = self.dataframe_gb[self.x_col][index]
            label_name = img_id[:-4]+'.txt'
            image = cv2.imread(path_img+img_id)

            if HorizontalFlip == False and VerticalFlip == False:
                aug = "_HorizontalFlip"
                transform = A.Compose([
                A.RandomCrop(width=image.shape[1], height=image.shape[0]),     
                A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))
        
            elif HorizontalFlip == True and VerticalFlip == False:
                aug = "_HorizontalFlip"
                transform = A.Compose([
                A.RandomCrop(width=image.shape[1], height=image.shape[0]),     
                A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))

            elif HorizontalFlip == False and VerticalFlip == True:
                aug = "_VerticalFlip"
                transform = A.Compose([
                A.RandomCrop(width=image.shape[1], height=image.shape[0]),     
                A.VerticalFlip(p=1)], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))
            else:
                aug = "_HVFlip"
                transform = A.Compose([
                A.RandomCrop(width=image.shape[1], height=image.shape[0]),     
                A.VerticalFlip(p=1),A.HorizontalFlip(p=1)], 
                bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))  
            ## อ่าน label
            label = open(labels_path+label_name,"r").read()
            class_labels  = (([i.split()[0] for i in label.splitlines()]))
            bboxes = (([i.split()[1:5] for i in label.splitlines()]))
            bboxes = [[float(i)  for i in bboxes[j]] for j in range(len(bboxes))]
            for i in range(len(class_labels)):
                bboxes[i].append(class_labels[i])
            
            transformed = transform(image=image, bboxes=bboxes,class_labels=class_labels) 
            transformed_image = transformed['image'] 
            transformed_bboxes = transformed['bboxes'] 
            transformed_class_labels = transformed['class_labels']
            cv2.imwrite(copyimage_path+img_id[:-4]+aug+".png",transformed_image)
            transformed_bboxes = [[i for i in transformed_bboxes[j]] for j in range(len(transformed_bboxes))]
            for i in range(len(transformed_bboxes)):
                transformed_bboxes[i][4] = int(transformed_bboxes[i][4])
            list_file = open(labels_path+label_name[:-4]+aug+".txt","w")
            for n_box in range(len(transformed_bboxes)):
                class_ = transformed_bboxes[n_box][4]
                x_cen = transformed_bboxes[n_box][0]
                y_cen = transformed_bboxes[n_box][1]
                w_cen = transformed_bboxes[n_box][2]
                h_cen = transformed_bboxes[n_box][3]
                list_file.write(
                            f"{class_} {x_cen} {y_cen} {w_cen} {h_cen}\n"
                        )
                
    def plot_image(self,index,line_thickness=5):
        label_ = list(self.label_.keys())
        path_img = self.dataframe_gb[self.img_path][index]
        img_id = self.dataframe_gb[self.x_col][index]
        img = Image.open(path_img+img_id).convert("RGB")

        records = self.dataframe[self.dataframe[self.x_col] == img_id]
        boxes = records[self.box_col].values
        labels = records[self.y_col].values

        for j in range(len(boxes)):
            box = boxes[j]
            label = labels[j]
            color = self.colors[self.label_[label]]

            draw = ImageDraw.Draw(img)
            line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
            draw.rectangle(tuple(box), width=line_thickness, outline=color)
            fontsize = max(round(max(img.size) / 40), 12)
            # font = ImageFont.truetype("/font/arial.ttf", fontsize)
            font = ImageFont.load_default()
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
            draw.text((box[0], box[1] - txt_height + 1), label, font=font)
            draw.text(((box[0], box[1] - txt_height + 1)), label,fill='white', font = font)

        img = np.asarray(img)
        plt.figure(figsize=(10,10))
        plt.imshow(img)

    def multi_plot_image(self,line_thickness=5):
        label_ = list(self.label_.keys())
        n = [random.randint(0, len(self.dataframe_gb)-1) for i in range(9)]
        plt.figure(figsize=(15,15))
        for i in range(9):
            plt.subplot(3,3,i+1)
            path_img = self.dataframe_gb[self.img_path][n[i]]
            img_id = self.dataframe_gb[self.x_col][n[i]]
            img = Image.open(path_img+img_id).convert("RGB")
            records = self.dataframe[self.dataframe[self.x_col] == img_id]
            boxes = records[self.box_col].values
            labels = records[self.y_col].values
            for j in range(len(boxes)):
                box = boxes[j]
                label = labels[j]
                color = self.colors[self.label_[label]]
                draw = ImageDraw.Draw(img)
                line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
                draw.rectangle(tuple(box), width=line_thickness, outline=color)
                fontsize = max(round(max(img.size) / 40), 12)
                # font = ImageFont.truetype("/font/arial.ttf", fontsize)
                font = ImageFont.load_default()
                txt_width, txt_height = font.getsize(label)
                draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
                draw.text((box[0], box[1] - txt_height + 1), label, font=font)
                draw.text(((box[0], box[1] - txt_height + 1)), label,fill='white', font = font)

            img = np.asarray(img)
            plt.imshow(img)