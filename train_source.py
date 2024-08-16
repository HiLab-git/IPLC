from utils import parse_config, set_random,niiDataset
from unet import UNet
from torch.utils.data import DataLoader
import torch
import matplotlib
import os
import argparse
from test_run import test
from metrics import dice_eval
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
matplotlib.use('Agg')
def get_data_loader(config,dataset,target):
    batch_size = config['train']['batch_size']
    data_root_mms = config['train']['data_root_mms']

    train_img = data_root_mms+'/train/img/{}'.format(target)
    train_lab = data_root_mms+'/train/lab/{}'.format(target)
    valid_img = data_root_mms+'/valid/img/{}'.format(target)
    valid_lab = data_root_mms+'/valid/lab/{}'.format(target)
    test_img = data_root_mms+'/test/img/{}'.format(target)
    test_lab = data_root_mms+'/test/lab/{}'.format(target)

    
    train_test = niiDataset(train_img,train_lab, dataset=dataset, target = target, phase = 'train')
    train_loader = DataLoader(train_test, batch_size = batch_size,shuffle=True, drop_last=True)
    val_dataset = niiDataset(valid_img,valid_lab, dataset=dataset, target = target, phase = 'valid')
    valid_loader = DataLoader(val_dataset, batch_size=1,shuffle=False, drop_last=False)
    test_dataset = niiDataset(test_img,test_lab, dataset=dataset, target = target, phase = 'test')
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False, drop_last=False)
    return train_loader,valid_loader,test_loader

def train(config, train_loader, valid_loader, test_loader, target, list_data, current_date, save_path):
    writer = SummaryWriter(
        log_dir=save_path + "/tensorboard/" + '/' + str(target) + '/' + current_date, comment='')
    directory_path = save_path + '/txt/' + str(target) + '/' + current_date
    file_path = os.path.join(directory_path, f'{target}.txt')
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(file_path, 'w') as file:
        file.write(current_date + "\n")
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    num_classes = config['network']['n_classes_mms']
    # load model
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    iplc_model = UNet(config).to(device)
    iplc_model.train()
    iplc_model.initialize()
    print("model initialize")
    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    j = 0
    best_dice = 0.
    for epoch in range(num_epochs):
        iplc_model.train()
        print('Epoch [%d/%d]' %(epoch, num_epochs))
        current_loss = 0.
        for i, (B, B_label, _,_) in tqdm(enumerate(train_loader)):
            B = B.to(device).detach()
            B_label = B_label.to(device).detach()
            loss_seg = iplc_model.train_source(B,B_label)
            current_loss += loss_seg
        loss_mean = current_loss / (i + 1)
        writer.add_scalar('loss', loss_mean, epoch)
        if (epoch) % valid_epochs == 0:
            current_dice = 0.
            with torch.no_grad():
                iplc_model.eval()
                for it,(xt,xt_label,xt_name,lab_Imag) in tqdm(enumerate(valid_loader)):
                    xt = xt.to(device)
                    xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                    output = iplc_model.test_with_name(xt)
                    output = output.squeeze(0)
                    output = torch.argmax(output,dim=1)        
                    output_ = output.cpu().numpy()
                    xt = xt.detach().cpu().numpy().squeeze()
                    output = output_.squeeze()
                    one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice,axis=0) 
                    current_dice += one_case_dice
            dice_mean = current_dice / (it + 1)
            writer.add_scalar('dice', dice_mean, epoch)
            if (current_dice / (it+1)) > best_dice:
                best_dice = current_dice / (it+1)
                model_dir = save_path + "/model/" + str(exp_name + '_' + target) + '/' + current_date
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), best_dice)
                torch.save(iplc_model.state_dict(), best_epoch)
                torch.save(iplc_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))
    iplc_model.update_lr()
    iplc_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=False)
    iplc_model.eval()
    test(config, iplc_model, valid_loader, test_loader, list_data, target, current_date, save_path)
    return list_data


def mian():
    # load config
    save_path = "train_source"
    current_date = time.strftime("%Y%m%d", time.localtime())
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train_source.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    list_data = []
    print(config)
    dataset = config['train']['dataset']
    for dataset in ['mms']:
        for target in ['B','C','D']:
            config['train']['dataset'] = dataset
            list_data.append(dataset)
            list_data.append(target)
            train_loader,valid_loader,test_loader = get_data_loader(config,dataset,target)
            list_data = train(config, train_loader, valid_loader, test_loader, target, list_data, current_date, save_path)
            directory_path = save_path + '/txt/' + str(target) + '/' + current_date
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            file_path = os.path.join(directory_path, f'{target}.txt')
            with open(file_path, 'w') as file:
                for line in list_data:
                    file.write(line + "\n")
        
if __name__ == '__main__':
    set_random()
    mian()