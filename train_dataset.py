import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from PIL import ImageEnhance
import numpy as np
import random
from PIL import ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
identity = lambda x:x
transformTypeDict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object): # 图像强化
    def __init__(self, transformDict):
        self.transforms = [(transformTypeDict[k], transformDict[k]) for k in transformDict]
        
    def __call__(self, img):
        out = img
        # 生成一个随机张量，用于决定图像强化的程度
        randTensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randTensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


class PILRandomGaussianBlur(object):

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
            
def get_color_distortion(s=0.5):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

    
class SetDataset:
    def __init__(self, data_path, num_class, batch_size):
        self.sub_meta = {}
        self.data_path = data_path
        self.num_class = num_class
        self.cl_list = range(self.num_class)

        print("==> Start to create sub meta data -- SetDataset")

        for cl in self.cl_list:
            self.sub_meta[cl] = []
        # ImageFolder 自动按照文件夹组织数据,并且自动分配类别标签

        print("==> Using ImageFolder dataset")
        img_data = ImageFolder(self.data_path)
        print("==> Finish ImageFolder dataset")

        for i, (path, label) in enumerate(img_data.imgs):
            # if i % 1000 == 0:
            #     print("==> Loading data: {}/{}".format(i, len(img_data)))
            #     print(path)
            self.sub_meta[label].append(path)


        # print("===> Test1")
        # for key, _ in self.sub_meta.items():
        #     print(len(self.sub_meta[key]))
        # print("===> Test2")

        self.sub_dataloader = []
        # 子数据集迭代器
        self.iter_sub_dataloader = []

        print("==> Start to create sub dataset -- SubDataset")

        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl],target=cl)
            dataloadr = torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            self.sub_dataloader.append(dataloadr)
            # 创建子数据集的迭代器
            self.iter_sub_dataloader.append(iter(dataloadr))
            # self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        print("==> Finish creating sub dataset -- SubDataset")

    def __getitem__(self, i):
        # 在下次迭代中将自动获取 sub_dataloader[i] 的下一个元素
        try:
            # 尝试获取当前迭代器的下一个元素
            data = next(self.iter_sub_dataloader[i])
        except StopIteration:
            # 当迭代器已经迭代完毕时，重新创建迭代器并获取第一个元素
            self.iter_sub_dataloader[i] = iter(self.sub_dataloader[i])
            data = next(self.iter_sub_dataloader[i])
        return data

    def __len__(self):
        return len(self.sub_dataloader)

# subdataset 子数据集
class SubDataset:
    def __init__(self, 
        sub_meta,
        target,
        size_crops=[224, 96],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14],
    ):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            # scale参数 指定裁剪区域的面积占原图像的面积的比例范围
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i], # 224 or 96
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            # 对不同组数据产生不同的裁剪结果
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        self.jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
        
        self.global_transforms = transforms.Compose([
                transforms.Resize([224,224]),
                ImageJitter(self.jitter_param),
                transforms.RandomHorizontalFlip(), # 随机水平翻转 以0.5的概率
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        
        self.sub_meta = sub_meta
        self.target = target
        
    def __getitem__(self,i):

        img = Image.open(self.sub_meta[i]).convert('RGB')
        # img = self.sub_meta[i]
        # 对原数据进行裁剪 并添加原图像
        # multi_crops将包含8张经过不同裁剪和处理的图片
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        raw_image = self.global_transforms(img)
        multi_crops.append(raw_image)
        
        return multi_crops, self.target # 返回8张图片和对应的标签


    def __len__(self):
        return len(self.sub_meta)
    
    
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class Eposide_DataManager(): # 轮次数据管理器
    def __init__(self, data_path, num_class, n_way=5, n_support=1, n_query=15, n_eposide=1):        
        super(Eposide_DataManager, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        print("==> Eposide_DataManager: %d-way %d-shot %d-query %d-eposide" % (n_way, n_support, n_query, n_eposide))

    def get_data_loader(self): 
        dataset = SetDataset(self.data_path, self.num_class, self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)  
        data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    
    np.random.seed(1111)
    data_path = r"E:\Datasets\miniImagenet\base"
    datamgr = Eposide_DataManager(data_path=data_path, num_class=2, n_way=2, n_support=1, n_query=2, n_eposide=2)
    base_loader = datamgr.get_data_loader()
    data = []
    for i, x in enumerate(base_loader):
        print(i)

    
    