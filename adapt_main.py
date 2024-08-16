from utils import parse_config, set_random, niiDataset
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
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace
from datetime import datetime
from ruamel.yaml import YAML

matplotlib.use('Agg')

def get_data_loader(config, dataset, target):
    batch_size = config['train']['batch_size']
    data_root_mms = config['train']['data_root_mms']
    train_img = data_root_mms + '/train/img/{}'.format(target)
    train_lab = data_root_mms + '/train/lab/{}'.format(target)
    valid_img = data_root_mms + '/valid/img/{}'.format(target)
    valid_lab = data_root_mms + '/valid/lab/{}'.format(target)
    test_img = data_root_mms + '/test/img/{}'.format(target)
    test_lab = data_root_mms + '/test/lab/{}'.format(target)

    train_test = niiDataset(train_img, train_lab, dataset=dataset, target=target, phase='train')
    train_loader = DataLoader(train_test, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = niiDataset(valid_img, valid_lab, dataset=dataset, target=target, phase='valid')
    valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_dataset = niiDataset(test_img, test_lab, dataset=dataset, target=target, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader


def train(config, train_loader, valid_loader, test_loader, target, list_data, save_path):
    # config SAM
    args = Namespace()
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    args.image_size = 256
    args.encoder_adapter = True
    args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"
    model = sam_model_registry["vit_b"](args).to(device)
    predictor = SammedPredictor(model)

    writer = SummaryWriter(
        log_dir=save_path + "/tensorboard/" + '/' + str(target), comment='')
    directory_path = save_path + '/txt/' + str(target)
    file_path = os.path.join(directory_path, f'{target}.txt')
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(file_path, 'w') as file:
        file.write('1' + "\n")

    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    sample_times = config['train']['sample_times']
    num_classes = config['network']['n_classes_mms']
    curve_weight = config['train']['curve_loss_weight']
    # load model
    iplc_model = UNet(config).to(device)
    iplc_model.train()

    iplc_model.load_state_dict(torch.load(config['train']['source_model_root_mms'], map_location='cpu'),
                                  strict=False)
    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    j = 0
    best_dice = 0.
    for epoch in range(num_epochs):
        iplc_model.train()
        print('Epoch [%d/%d]' % (epoch, num_epochs))
        current_diceloss = 0
        current_curve_loss = 0
        for i, (B, B_label, B_name,lab_Imag) in tqdm(enumerate(train_loader)):
            B = B.to(device).detach()
            # B_label = B_label.to(device).detach()
            iplc_model.generate_sam_pl(B, B_name, predictor, save_path, num_classes, sample_times, target)
            diceloss, curve_loss = iplc_model.domain_adaptation(B, curve_weight)
            current_diceloss += diceloss
            current_curve_loss += curve_loss
        diceloss_mean = current_diceloss / (i + 1)
        curve_loss_mean = current_curve_loss / (i + 1)
        writer.add_scalar('diceloss_mean', diceloss_mean, epoch)
        writer.add_scalar('curve_loss_mean', curve_loss_mean, epoch)
        if (epoch) % valid_epochs == 0:
            current_dice = 0.
            with torch.no_grad():
                iplc_model.eval()
                for it, (xt, xt_label, xt_name, lab_Imag) in tqdm(enumerate(valid_loader)):
                    xt = xt.to(device)
                    xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                    output = iplc_model.test_with_name(xt)
                    output = output.squeeze(0)
                    output = torch.argmax(output, dim=1)
                    output_ = output.cpu().numpy()
                    xt = xt.detach().cpu().numpy().squeeze()
                    output = output_.squeeze()
                    one_case_dice = dice_eval(output, xt_label, num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice, axis=0)
                    current_dice += one_case_dice
            dice_mean = current_dice / (it + 1)
            writer.add_scalar('dice', dice_mean, epoch)
            if (current_dice / (it + 1)) > best_dice:
                best_dice = current_dice / (it + 1)
                model_dir = save_path + "/model/" + str(exp_name + '_' + target)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), best_dice)
                torch.save(iplc_model.state_dict(), best_epoch)
                torch.save(iplc_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))
    iplc_model.load_state_dict(torch.load(best_epoch, map_location='cpu'), strict=False)
    iplc_model.eval()
    test(config, iplc_model, valid_loader, test_loader, list_data, target, save_path)
    return list_data



def mian():

    save_path_source = "IPLC"
    now = datetime.now()
    save_path = os.path.join(save_path_source, now.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/adapt.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    list_data = []
    print(config)
    # Create the YAML object
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    # Write the config to a YAML file
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    for dataset in ['mms']:
        for target in ['B', 'C', 'D']:
            config['train']['dataset'] = dataset
            list_data.append(dataset)
            list_data.append(target)
            train_loader, valid_loader, test_loader = get_data_loader(config, dataset, target)
            list_data = train(config, train_loader, valid_loader, test_loader, target, list_data,
                              save_path)
            directory_path = save_path + '/txt/' + str(target)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            file_path = os.path.join(directory_path, f'{target}.txt')
            with open(file_path, 'w') as file:
                for line in list_data:
                    file.write(line + "\n")


if __name__ == '__main__':
    set_random()
    mian()