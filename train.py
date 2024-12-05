from torch.utils.data import DataLoader
import os
import argparse
import torch
import time
import HDPNet_model
import dataset
import loss
import pandas as pd
from tqdm import tqdm
from utils import one_cycle, intersect_dicts
from torch.optim import lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser('--HDPNet--')
    parser.add_argument('--base_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--path', type=str, help='path to train dataset')
    parser.add_argument('--pretrain', type=str, help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')

    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--epochs', default=150, type=int,
                        help='number of training epochs')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()
    return args


def main(args):
    ### model ###
    net = HDPNet_model.Model(args.pretrain, img_size=384)

    # root = '/media/pc/1CAC3A59AC3A2E20/HDPNet/'
    # ckpt_path = 'hourglasspvt/fitune_all/pvt1_camo+cod/model_55_loss_0.10269.pth'
    # net = HDPNet_model.Model(None, img_size=384)
    # ckpt = torch.load(root + ckpt_path, map_location='cpu')
    # csd = intersect_dicts(ckpt, net.state_dict())  # intersect
    # msg = net.load_state_dict(csd, strict=False)  # load
    #
    # print("====================================")
    # pt_name = ckpt_path.split('/')[-1]
    # print(f'Transferred {len(csd)}/{len(ckpt)} items from {pt_name}')

    net.cuda()
    encoder_param = []
    decoer_param = []
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    optimizer = torch.optim.Adam(
        [{"params": encoder_param, "lr": args.base_lr * 0.1}, {"params": decoer_param, "lr": args.base_lr}])

    # Scheduler
    # lf = one_cycle(1, 0.01, args.epochs)  # cosine 1->hyp['lrf']
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    ### resume training if necessary ###
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    ### Fine tuning for MoCA ###
    if args.ft_for_MoCA is not None:
        ckpt = torch.load(args.ft_for_MoCA, map_location='cpu')
        net.load_state_dict(ckpt)
        print("Fine tuning for MoCA, ckpt from: {}".format(args.ft_for_MoCA))

    ### data ###
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size_per_gpu, num_workers=args.batch_size_per_gpu,
                            collate_fn=dataset.my_collate_fn, drop_last=True, shuffle=True)

    # torch.backends.cudnn.benchmark = True

    ### main loop ###
    # scheduler.last_epoch = - 1  # do not move
    epochs = args.epochs
    loss_result_curve = []
    t1 = time.time()
    for curr_epoch in range(0, epochs + 1):
        if curr_epoch == 50 or curr_epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print("Learning rate:", param_group['lr'])

        net.train()
        running_loss_all, running_loss_m = 0., 0.
        count = 0
        with tqdm(Dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {curr_epoch + 1}")
                count += 1
                img, label = data['img'].cuda(), data['label'].cuda()
                out = net(img)  # [b,1,384,384]
                all_loss, loss_m = loss.multi_bce(out, label)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                running_loss_all += all_loss.item()
                running_loss_m += loss_m

                tepoch.set_postfix(loss_all=running_loss_all / count, loss_main=running_loss_m / count)
        # scheduler.step()
        if curr_epoch % 2 == 0:
            ckpt_save_root = "/share/home/project/HDPNet/results/ckpt_save"
            if not os.path.exists(ckpt_save_root):
                os.mkdir(ckpt_save_root)
            torch.save(net.state_dict(),
                       ckpt_save_root + "/model_{}_loss_{:.5f}.pth".format(curr_epoch, running_loss_all / count)
                       )
        loss_result_curve.append(running_loss_all / count)

    t2 = time.time()
    total_time = t2 - t1
    print('trianing time：{:0>2}:{:0>2}'.format(int(total_time // 3600), int((total_time % 3600) // 60)))

    df = pd.DataFrame(loss_result_curve)
    df.to_excel(os.path.join(ckpt_save_root, 'HDPNet-Epoches_' + str(epochs) + '.xlsx'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
