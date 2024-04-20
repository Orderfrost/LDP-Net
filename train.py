import sys

import train_dataset # train
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import ProtoNet
import torch.nn as nn
from torch.autograd import Variable
import utils
import random
import copy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params):
    model.train()
    top1 = utils.AverageMeter()
    total_loss = 0
    softmax = torch.nn.Softmax(dim=1)
    eps = 1e-7
    for i, batch in tqdm(enumerate(train_loader)):
        with torch.autograd.set_detect_anomaly(True):
            print("===> %d episode" % (i+1))
            optimizer.zero_grad()
            x, y = batch
            print("type of x: ", type(x))
            print("type of y: ", type(y))
            print("y.size : ", y.size())

            #x_224 = torch.stack(x[:2]).cuda() # (2,way,shot+query,3,224,224)
            # torch.stack() 将一个list的tensor拼接成一个tensor 如将3个(2,3,4)的tensor拼接成(3,2,3,4)
            x_96 = torch.stack(x[2:8]).cuda() # (6,way,shot+query,3,96,96)
            x_224 = torch.stack(x[8:]).cuda() # (1,way,shot+query,3,224,224)
            support_set_anchor = x_224[0,:,:params.n_support,:,:,:] # (way,shot,3,224,224)
            query_set_anchor = x_224[0,:,params.n_support:,:,:,:] # (way,query,3,224,224)
            query_set_aug_96 = x_96[:,:,params.n_support:,:,:,:] # (6,way,query,3,96,96)
            temp_224 = torch.cat((support_set_anchor, query_set_anchor), 1) # (way,shot+query,3,224,224)
            # contiguous() 返回一个具有相同数据但是不同尺寸的tensor，深拷贝
            temp_224 = temp_224.contiguous().view(params.n_way*(params.n_support+params.n_query),3,224,224) # (way*(shot+query),3,224,224)

            # 进入特征提取  计算原型 ResNet10
            temp_224 = model(temp_224) # (way*(shot+query),512)
            # print("model output : ", temp_224[:2])
            print("==> Features extracted <==")

            temp_224 = temp_224.view(params.n_way, params.n_support+params.n_query, 512) # (way,shot+query,512)
            # 支持集原型
            support_set_prototype = temp_224[:,:params.n_support,:] # (way,shot,512)
            support_set_prototype = torch.mean(support_set_prototype, 1) # (way, 512)
            # 询问集
            query_set_anchor = temp_224[:,params.n_support:,:] # (way,query,512)
            #
            query_set_anchor = query_set_anchor.contiguous().view(params.n_way*params.n_query, 512).unsqueeze(0) # (1,way*query,512)

            query_set_aug_96 = query_set_aug_96.contiguous().view(6*params.n_way*params.n_query,3,96,96)# (6*way*query,3,96,96)
            with torch.no_grad():
                query_set_aug_96 = Siamese_model(query_set_aug_96) # (6*way*query,512)
            query_set_aug_96 = query_set_aug_96.view(6, params.n_way*params.n_query, 512) # (6, 5*15, 512)
            query_set = torch.cat((query_set_anchor, query_set_aug_96), 0) # (7, 5*15, 512)
            query_set = query_set.contiguous().view(7*params.n_way*params.n_query, 512) # (7*5*15, 512)


            pred_query_set = head(support_set_prototype, query_set) # (7*5*15,5)
            print("==> Distance between prototype and queryset <==")

            pred_query_set = pred_query_set.contiguous().view(7, params.n_way*params.n_query, params.n_way) # (7,75,5)
            # 未增强的查询集与原型 距离
            pred_query_set_anchor = pred_query_set[0] # (75,5)

            # 增强后的查询集与原型 距离
            pred_query_set_aug = pred_query_set[1:] # (6,75,5)

            query_set_y = torch.from_numpy(np.repeat(range(params.n_way), params.n_query)) # (75,)
            # query_set_y = Variable(query_set_y.cuda())
            query_set_y = query_set_y.cuda()
            ce_loss = loss_fn(pred_query_set_anchor, query_set_y.long())


            if torch.isnan(ce_loss):
                print("Detecet ce_loss NaN")
            else:
                print("ce_loss: ", ce_loss)

            print("==> Ce_loss Computed <==")

            pred_query_set_anchor = softmax(pred_query_set_anchor) # (75,5)
            pred_query_set_aug = pred_query_set_aug.contiguous().view(6*params.n_way*params.n_query, params.n_way) #(6*75,5)
            pred_query_set_aug = softmax(pred_query_set_aug)
            pred_query_set_anchor = torch.cat([pred_query_set_anchor for _ in range(6)], dim=0) # (6*75,5)

            # self_image_loss = loss_fn(pred_query_set_aug, pred_query_set_anchor) / 6
            self_image_loss = torch.mean(torch.sum((-pred_query_set_aug)*torch.log(pred_query_set_anchor+eps), dim=1))
            # self_image_loss = torch.mean(torch.sum((-pred_query_set_anchor)*torch.log(pred_query_set_aug)+eps, dim=1))

            if torch.isnan(self_image_loss):
                print("Detecet self_image_loss NaN")
            else:
                print("self_image_loss: ", self_image_loss)
            print("==> Self_image_loss Computed <==")


            pred_query_set_global = pred_query_set[0] # (75,5)
            pred_query_set_global = pred_query_set_global.view(params.n_way, params.n_query, params.n_way)

            rand_id_global = np.random.permutation(params.n_query)
            pred_query_set_global = pred_query_set_global[:, rand_id_global[0], :] # (way,way)
            pred_query_set_global = softmax(pred_query_set_global) # (way,way)
            pred_query_set_global = pred_query_set_global.unsqueeze(0) # (1,5,5)
            pred_query_set_global = pred_query_set_global.expand(6, params.n_way, params.n_way) # (6,5,5)
            pred_query_set_global = pred_query_set_global.contiguous().view(6*params.n_way, params.n_way) # (6*way,way)


            rand_id_local_sample = np.random.permutation(params.n_query)
            pred_query_set_local = pred_query_set_aug.view(6, params.n_way, params.n_query, params.n_way)
            pred_query_set_local = pred_query_set_local[:, :, rand_id_local_sample[0], :] # (6,way,way)
            pred_query_set_local = pred_query_set_local.contiguous().view(6*params.n_way, params.n_way) # (6*way,way)
            pred_query_set_local = softmax(pred_query_set_local) # (6*way,way)

            # cross_image_loss = loss_fn(pred_query_set_local, pred_query_set_global) / 6
            cross_image_loss = torch.mean(torch.sum((-pred_query_set_local)*torch.log(pred_query_set_global+eps), dim=1))
            # cross_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_local**(-pred_query_set_global)), dim=1))

            if torch.isnan(cross_image_loss):
                print("Detecet cross_image_loss NaN")
            else:
                print("cross_image_loss: ", cross_image_loss)
            print("==> cross_image_loss Computed <==")

            loss = ce_loss + self_image_loss  * params.lamba1 + cross_image_loss * params.lamba2
            if torch.isnan(loss):
                print("Detecet loss NaN")
            else:
                print("loss: ", loss)
            print("==> cross_image_loss Computed <==")
            _, predicted = torch.max(pred_query_set[0].data, 1) # (75,)
            correct = predicted.eq(query_set_y.data).cpu().sum()
            if i%5 == 0:
                print("query_set_y.data: ", query_set_y.data)
                print("predicted: ", predicted.data)
            # print("query_set_y.size(0)", query_set_y.size(0))
            top1.update(correct.item()*100 / (query_set_y.size(0)+0.0), query_set_y.size(0))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(model.parameters(), Siamese_model.parameters()):

                param_k.data = param_k.data * params.m + param_q.data * (1. - params.m)
    
        total_loss = total_loss + loss.item()
        # sys.exit(0)

    avg_loss = total_loss/float(i+1)
    print(avg_loss, top1.avg)
    print("==> Epoch: {}, Avg Loss: {:.4f}, Avg Acc: {:.2f}".format(epoch+1, avg_loss, top1.avg))

    return avg_loss, top1.avg
        
 
                
if __name__=='__main__':

    params = parse_args_eposide_train()

    setup_seed(params.seed)

    print("==> Preparing data...")
    datamgr_train = train_dataset.Eposide_DataManager(data_path=params.source_data_path, num_class=params.train_num_class, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.train_n_eposide)
    train_loader = datamgr_train.get_data_loader()

    print("==> Building model...")
    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)

    head = ProtoNet.ProtoNet()

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    print("==> Loading pretrain model...")
    # tmp = torch.load(params.pretrain_model_path)
    # state = tmp['state']
    # model.load_state_dict(state)
    Siamese_model = copy.deepcopy(model)
    model = model.cuda()
    Siamese_model = Siamese_model.cuda()
    head = head.cuda()

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model.parameters()}], lr=params.lr)

    print("==> Start training...")
    for epoch in range(params.epoch):
        print('==> Epoch:', epoch+1)
        train_loss, train_acc = train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params)
        print('train:', epoch+1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc)
    outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch+1))
    torch.save({
    'epoch':epoch+1, 
    'state_model':model.state_dict(),
    'state_Siamese_model':Siamese_model.state_dict()},
     outfile) 


    
    
    
    