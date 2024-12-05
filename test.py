import torch
import HDPNet_model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
from thop import profile
from tqdm import tqdm


if __name__ =='__main__':
    batch_size = 1
    net = HDPNet_model.Model(None, img_size=384).cuda()

    root_path = '/share/home/project/HDPNet/'
    ckpt = 'ckpt_save/model_108_loss_0.02126_best.pth'

    Dirs = ["/share/home/dataset/TestDataset/CAMO",
            "/share/home/dataset/TestDataset/COD10K",
            "/share/home/dataset/TestDataset/CHAMELEON",
            "/share/home/dataset/TestDataset/NC4K"]

    result_save_root = root_path+"results/"

    for m in ckpt:
        print(m)
        pretrained_dict = torch.load(root_path + m)
        net.load_state_dict(pretrained_dict)

        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists(result_save_root):
                os.mkdir(result_save_root)
            Dataset = dataset.TestDataset(Dir, 384)
            Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            count=0
            with tqdm(Dataloader, unit="batch") as tepoch:
                for data in tepoch:
                    count+=1
                    img, label = data['img'].cuda(), data['label'].cuda()
                    name = data['name'][0].split("/")[-1]
                    with torch.no_grad():
                        out = net(img)[2]
                    B,C,H,W = label.size()
                    o = F.interpolate(out, (H, W), mode='bilinear', align_corners=True).detach().cpu().numpy()[
                        0, 0]  # [H,W]
                    o = (o-o.min())/(o.max()-o.min()+1e-8)
                    o = (o*255).astype(np.uint8)
                    imwrite(result_save_root+Dir.split("/")[-1]+"/"+name, o)
    print("Test finished!")

