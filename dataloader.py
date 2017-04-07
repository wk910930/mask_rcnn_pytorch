import os
import sys
import collections
import torch
from torch.utils.data.dataloader import DataLoader, DataLoaderIter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


def getDataloaders(data, config_of_data, splits=['train', 'val'],
                   data_root='data', batch_size=16, normalized=True,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None

    if data.find('coco') >= 0:
        print('loading ' + data)
        print(config_of_data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        common_trans = [transforms.ToTensor()]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        compose = transforms.Compose(common_trans)

        # uses last 5000 images of the original val split as the
        # mini validation set
        trainval_set = CocoDetectionTrainVal(data_root, scale_size=800, transform=compose)
        if 'train' in splits:
            if config_of_data['train_split'] == 'train':
                indices = range(trainval_set.train_len())
            elif config_of_data['train_split'] == 'trainval35k':
                indices = range(len(trainval_set) - 5000)
            else:
                raise NotImplementedError
            train_loader = DataLoader(
                trainval_set, batch_size=batch_size,
                collate_fn=coco_collate,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                num_workers=num_workers, pin_memory=False)
        if 'val' in splits:
            if config_of_data['val_split'] == 'val':
                indices = range(trainval_set.train_len(), len(trainval_set))
            elif config_of_data['val_split'] == 'minival':
                indices = range(len(trainval_set) - 5000, len(trainval_set))
            else:
                raise NotImplementedError
            val_loader = DataLoader(
                trainval_set, batch_size=batch_size,
                collate_fn=coco_collate,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                num_workers=num_workers, pin_memory=False)
        if 'test' in splits:
            # TODO: for loading testing files
            raise NotImplementedError
    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader


# Based on CocoDetection in torchvision
class CocoDetectionTrainVal(torch.utils.data.Dataset):

    def __init__(self, root, scale_size=None, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        ann_train = os.path.join(self.root, 'annotations/instances_train2014.json')
        ann_val = os.path.join(self.root, 'annotations/instances_val2014.json')
        self.coco_train = COCO(ann_train)
        self.coco_val = COCO(ann_val)
        self.ids = list(self.coco_train.imgs.keys()) + list(self.coco_val.imgs.keys())
        self.transform = transform
        self.scale_size = scale_size
        self.ord2cid = sorted(self.coco_train.cats.keys())
        self.cid2ord = {i: o for o, i in enumerate(self.ord2cid)}

    def __getitem__(self, index):
        coco = self.coco_train if index < len(self.coco_train.imgs) else self.coco_val
        img_root = 'train2014' if index < len(self.coco_train.imgs) else 'val2014'
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, img_root, path)).convert('RGB')

        for ann in anns:
            # COCO uses x, y, w, h, but Faster RCNN uses x1, y1, x2, y2
            ann['bbox'][2] += ann['bbox'][0]
            ann['bbox'][3] += ann['bbox'][1]
            ann['ordered_id'] = self.cid2ord[ann['category_id']]
            ann['scale_ratio'] = 1.
            ann['mask'] = torch.from_numpy(coco.annToMask(ann)).float().unsqueeze(0)

        # scaling
        if self.scale_size is not None:
            w, h = img.size
            scale_ratio = self.scale_size / w if w < h else self.scale_size / h
            if scale_ratio != 1.:
                img = img.resize((int(w * scale_ratio), int(h * scale_ratio)),
                                 Image.BILINEAR)
                for ann in anns:
                    ann['area'] *= scale_ratio**2
                    ann['bbox'] = [x * scale_ratio for x in ann['bbox']]
                    ann['segmentation'] = [[x * scale_ratio for x in y]
                                           for y in ann['segmentation']]
                    mask = transforms.ToPILImage()(ann['mask'])
                    mask = mask.resize((round(w * scale_ratio + 0.5),
                                        round(h * scale_ratio + 0.5)),
                                       Image.BILINEAR)
                    ann['mask'] = transforms.ToTensor()(mask)
                    ann['scale_ratio'] = scale_ratio

        if self.transform is not None:
            img = self.transform(img)


        return img, anns

    def __len__(self):
        return len(self.ids)

    def train_len(self):
        return len(self.coco_train.imgs)

    def val_len(self):
        return len(self.coco_val.imgs)


def coco_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size, or put collade recursively for dict"
    if isinstance(batch[0], tuple):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [coco_collate(samples) for samples in transposed]
    return batch
