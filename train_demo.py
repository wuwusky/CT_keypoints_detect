import random
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from models import mobilenet_v3_small, mobilenet_v3_large
import skimage.transform as trans
import torch.optim as optim
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_crop(temp_data):
    if random.random()<0.5:
        tx = random.randint(0,100)
        ty = random.randint(0,100)
    else:
        tx = 0
        ty = 0

    temp_data_crop = temp_data[:,tx:,ty:]
    return temp_data_crop, tx, ty

def random_move_z(temp_data):
    if random.random()<0.5:
        tz = random.randint(-50, 10)
        if tz > 0:
            temp_data_moved = temp_data[tz:,:,:]
        elif tz < 0:
            temp_zeros_head = np.zeros_like(temp_data)[:tz,:,:]
            temp_data_moved = np.concatenate([temp_zeros_head, temp_data[:-tz,:,:]])
        else:
            temp_data_moved = temp_data.copy()
    else:
        temp_data_moved = temp_data.copy()
        tz = 0
    return temp_data_moved, tz



def random_shape_aug(shape):
    temp_z, tempx, tempy = shape
    r_z = random.choice([100,150,temp_z])
    r_x = random.choice([256,320,tempx])
    r_z = r_x
    return (r_z, r_x, r_z)

def random_bright_contrast_aug(temp_data):
    if random.random() < 0.5:
        delta = random.uniform(-0.2, 0.2)
        temp_data += delta
        temp_data = np.clip(temp_data, a_min=0, a_max=1)
    
    if random.random() < 0.5:
        alpha = random.uniform(0.75, 1.25)
        temp_data *= alpha
        temp_data = np.clip(temp_data, a_min=0, a_max=1)
    
    return temp_data


class Dataset_ct_demo(Dataset):
    def __init__(self, list_data, status='valid', shape=(200, 512, 512)):
        super(Dataset_ct_demo, self).__init__()
        self.list_data = list_data
        self.status = status
        self.shape = shape
        


    
    def __getitem__(self, index):

        if self.status == 'train':
            temp_shape = random_shape_aug(self.shape)
        else:
            temp_shape = self.shape

        temp_data = np.load(self.list_data[index][0])
        temp_label_info = self.list_data[index][1]

        # tx = int(float(temp_label_info[5][0]))
        # ty = int(float(temp_label_info[5][1]))
        # tz = int(temp_label_info[5][2])
        # temp_show = np.uint8(temp_data[tz]*255)
        # temp_show = cv2.cvtColor(temp_show, cv2.COLOR_GRAY2BGR)
        # temp_show = cv2.circle(temp_show, (tx, ty), 5, (0,255,0), -1)
        # cv2.imshow('show', temp_show)
        # cv2.waitKey()

        
        if self.status == 'train':
            temp_data, tx, ty = random_crop(temp_data)
            temp_data = random_bright_contrast_aug(temp_data)
            temp_data, tz = random_move_z(temp_data)
        else:
            tx = 0
            ty = 0
            tz = 0
        num_seq, h, w = temp_data.shape
        temp_data = trans.resize(temp_data, temp_shape)

        
        
        temp_label_location = []
        temp_label_slice = []
        temp_label_slice_id = [0]*temp_shape[0]
        for temp_info in temp_label_info:
            tempx = (float(temp_info[0])-tx)/w
            tempy = (float(temp_info[1])-ty)/h
            tempid = (float(temp_info[2])-tz)/num_seq
            try:
                temp_index = int(temp_shape[0]*tempid)
                if temp_index>temp_shape[0]-1:
                    temp_index = temp_shape[0]-1

                # # ########raondom flip/x,y,z
                # if self.status == 'train':
                #     if random.random() < 0.5: ##Z-axis
                #         temp_data = np.flip(temp_data, 0)

                #         tempid = 1-tempid
                #         temp_index = temp_shape[0]-temp_index

                #     if random.random() < 0.5: ## up-down
                #         temp_data = np.flip(temp_data, 1)
                #         temp_data = np.flip(temp_data, 2)

                #         tempy = 1-tempy
                #         tempx = 1-tempx
                        

                        

                temp_label_location += [tempx, tempy]
                temp_label_slice.append(tempid)
                temp_label_slice_id[temp_index] = 1

            except Exception as e:
                print(e)
                print(temp_info)
                print(self.list_data[index][0])

        temp_label_location = np.array(temp_label_location)
        temp_label_slice = np.array(temp_label_slice)
        temp_label_slice_id = np.array(temp_label_slice_id)

        
                

            ### crop 
            #### ... ###



        temp_data = np.expand_dims(temp_data, axis=1)
        vecs_tensor = torch.from_numpy(temp_data.copy()).float()

        label_tensor_location = torch.from_numpy(temp_label_location).float()
        label_tensor_slice = torch.from_numpy(temp_label_slice).float()
        label_tensor_slice_id = torch.from_numpy(temp_label_slice_id).float()

        return vecs_tensor, label_tensor_location, label_tensor_slice, label_tensor_slice_id
    
    def __len__(self):
        return len(self.list_data)
    

def get_label_info(info_dir):
    f = open(info_dir, mode='r', encoding='utf-8')
    info_lines = f.readlines()
    info_label = []
    for info_line in info_lines[2:13]:
        info_line = info_line.strip('\n').split(',')
        info_label.append([info_line[-3], info_line[-2], info_line[-1]])
    return info_label

import time
import torch.nn.functional as F
def main_train():
    max_epoch = 100
    learn_rate = 1e-4
    batch_size = 1
    step_optim = 8

    data_root_dir = './data_np_train/'
    json_root_dir = './label_csv/'
    list_names = os.listdir(data_root_dir)

    list_data_train = []
    list_data_valid = []
    for i, temp_name in enumerate(tqdm(list_names, ncols=200)):
        temp_data_dir = data_root_dir + temp_name
        try:
            temp_label_info = get_label_info(json_root_dir + temp_name[:-3]+'xlsx.csv')
            if (i+1)%10 == 0:
                list_data_valid.append([temp_data_dir, temp_label_info])
            else:
                list_data_train.append([temp_data_dir, temp_label_info])
        except Exception as e:
            continue

    print('train sample num:', len(list_data_train))
    print('valid sample num:', len(list_data_valid))

    dataset_train = Dataset_ct_demo(list_data_train[:], status='train', shape=(200, 320, 320))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)

    dataset_valid = Dataset_ct_demo(list_data_valid, status='valid', shape=(200, 320, 320))
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True)

    
    try:
        model_ct_demo = torch.load('./demo_ct.pth')
    except Exception as e:
        model_ct_demo = mobilenet_v3_small()


    model_ct_demo = model_ct_demo.to(device)
    optimizer = optim.Adam(model_ct_demo.parameters(), lr=learn_rate)
    # optimizer = optim.AdamW(model_ct_demo.parameters(), lr=learn_rate, weight_decay=1e-4, amsgrad=True)
    lr_sch = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch//2], gamma=0.1)
    loss_fun = nn.SmoothL1Loss()
    loss_att = nn.BCELoss()
    # loss_fun = nn.MSELoss()
    # loss_fun = nn.L1Loss()
    temp_best_mae = 10000

    for epoch in range(max_epoch):
        time_start = time.time()
        model_ct_demo.train()
        for i, data in enumerate(loader_train):
            datas, label_location, label_slice, label_slice_id = data
            datas, label_location, label_slice, label_slice_id = datas.to(device), label_location.to(device), label_slice.to(device), label_slice_id.to(device)

            out_slice_id, out_location, out_slice = model_ct_demo.forward_mil(datas)
            loss_id = loss_fun(out_slice_id, label_slice_id)
            loss_location = loss_fun(out_location, label_location)
            loss_slice = loss_fun(out_slice, label_slice)


            loss_target_location = loss_fun(out_location*512, label_location*512)
            loss_target_slice = loss_fun(out_slice*512, label_slice*512)

            loss_1 = loss_id + loss_location + loss_slice
            loss_target = loss_target_location + loss_target_slice

            loss = loss_1*10.0 + loss_target
            loss.backward()

            if (i+1) % step_optim == 0:
                optimizer.step()
                optimizer.zero_grad()

            tempshape = datas.shape[-1]
            if i<5:
                print('Epoch:{}/{}, Iter:{}/{}, loss:{:.4f}, loss_target:{:.4f}, lr:{:.6f}, tempshape:{}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss_1.item(), loss_target.item(), optimizer.param_groups[0]['lr'], tempshape))
            else:
                print('Epoch:{}/{}, Iter:{}/{}, loss:{:.4f}, loss_target:{:.4f}, lr:{:.6f}, tempshape:{}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss_1.item(), loss_target.item(), optimizer.param_groups[0]['lr'], tempshape), end='\r')

        lr_sch.step()
        model_ct_demo.eval()
        torch.save(model_ct_demo,  './demo_ct.pth')

        time_cost = time.time() - time_start
        print('')
        print('Epoch cost time:{:.2f}s'.format(time_cost))


        list_mae = []
        for i, data in enumerate(tqdm(loader_valid, ncols=200)):
            datas, label_location, label_slice, label_slice_id = data
            datas = datas.to(device)

            with torch.no_grad():
                out_slice_id, out_location, out_slice = model_ct_demo.forward_mil(datas)
            # out = out_location[0].cpu().numpy().tolist()
            # out_slice = out_slice[0].cpu().numpy().tolist()
            # labels = label_location[0].cpu().numpy().tolist()
            # labels_id = label_slice[0].cpu().numpy().tolist()

            out_all = torch.cat([out_location*512, out_slice*200], dim=1).cpu()
            label_all = torch.cat([label_location*512, label_slice*200], dim=1).cpu()
            temp_loss = F.l1_loss(out_all, label_all)
            list_mae.append(temp_loss.item())
            
            # for pr, gt in zip(out, labels):
            #     print('{:.4f},{:.4f}'.format(pr, gt))
            
            # for pr, gt in zip(out_slice, labels_id):
            #     print('{:.4f},{:.4f}'.format(pr, gt))

            # break
        avg_mae = np.mean(list_mae)
        if avg_mae <= temp_best_mae:
            temp_best_mae = avg_mae
            torch.save(model_ct_demo, './demo_ct_best.pth')

        print('valid_mae:{:.4f}, best_mae:{:.4f}'.format(avg_mae, temp_best_mae))


def demo_eval():
    data_root_dir = './data_np_train/'
    json_root_dir = './label_csv/'
    list_names = os.listdir(data_root_dir)

    list_data_train = []
    list_data_valid = []
    for i, temp_name in enumerate(tqdm(list_names, ncols=200)):
        temp_data_dir = data_root_dir + temp_name
        try:
            temp_label_info = get_label_info(json_root_dir + temp_name[:-3]+'xlsx.csv')
            if (i+1)%10 == 0:
                list_data_valid.append([temp_data_dir, temp_label_info])
            else:
                list_data_train.append([temp_data_dir, temp_label_info])
        except Exception as e:
            continue

    print('train sample num:', len(list_data_train))
    print('valid sample num:', len(list_data_valid))

    dataset_train = Dataset_ct_demo(list_data_train[:], status='valid', shape=(200, 320, 320))
    loader_train = DataLoader(dataset_train, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)

    dataset_valid = Dataset_ct_demo(list_data_valid, status='valid', shape=(200, 320, 320))
    loader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)

    
    model_ct_demo = torch.load('./demo_ct_best.pth', map_location='cpu')
    model_ct_demo = model_ct_demo.to(device)
    model_ct_demo.eval()


    time_start = time.time()

    list_mae = []
    for i, data in enumerate(tqdm(loader_valid, ncols=200)):
        datas, label_location, label_slice, label_slice_id = data
        datas = datas.to(device)

        with torch.no_grad():
            out_slice_id, out_location, out_slice = model_ct_demo.forward_mil(datas)

        out_all = torch.cat([out_location*512, out_slice*200], dim=1).cpu()
        label_all = torch.cat([label_location*512, label_slice*200], dim=1).cpu()
        temp_loss = F.l1_loss(out_all, label_all)
        list_mae.append(temp_loss.item())
        

    avg_mae = np.mean(list_mae)

    print('valid_mae:{:.4f}'.format(avg_mae))
    time_cost = time.time() - time_start
    print('Epoch cost time:{:.2f}s'.format(time_cost))

from demo_data_preprocess import get_imageData_with_real

def demo_model_predict():
    data_root_dir = './data_np_train/'
    json_root_dir = './label_csv/'
    list_names = os.listdir(data_root_dir)

    list_data_train = []
    list_data_valid = []
    for i, temp_name in enumerate(tqdm(list_names, ncols=200)):
        temp_data_dir = data_root_dir + temp_name
        try:
            temp_label_info = get_label_info(json_root_dir + temp_name[:-3]+'xlsx.csv')
            if (i+1)%10 == 0:
                list_data_valid.append([temp_data_dir, temp_label_info])
            else:
                list_data_train.append([temp_data_dir, temp_label_info])
        except Exception as e:
            continue

    print('train sample num:', len(list_data_train))
    print('valid sample num:', len(list_data_valid))


    
    model_ct_demo = torch.load('./demo_ct_best.pth', map_location='cpu')
    model_ct_demo = model_ct_demo.to(device)
    model_ct_demo.eval()


    time_start = time.time()

    list_mae = []
    for i, temp_data in enumerate(tqdm(list_data_valid, ncols=200)):

        
        datas, label_location, label_slice, label_slice_id = data
        datas = datas.to(device)

        with torch.no_grad():
            out_slice_id, out_location, out_slice = model_ct_demo.forward_mil(datas)

        out_all = torch.cat([out_location*512, out_slice*200], dim=1).cpu()
        label_all = torch.cat([label_location*512, label_slice*200], dim=1).cpu()
        temp_loss = F.l1_loss(out_all, label_all)
        list_mae.append(temp_loss.item())
        

    avg_mae = np.mean(list_mae)

    print('valid_mae:{:.4f}'.format(avg_mae))
    time_cost = time.time() - time_start
    print('Epoch cost time:{:.2f}s'.format(time_cost))




if __name__ == '__main__':
    # main_train()
    demo_eval()
    # demo_model_predict()

