from models.resnet101 import ResNet101
from models.my_model import MyGCN
from config import opt
from preprocess_and_load.dataset import ChestXrayDataSet
from torch.utils.data import DataLoader
import torch
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(len(opt.classes)):
        tmp = np.sum(gt_np[:, i], axis=0)
        if tmp == 0:
            AUROCs.append(0)
            continue
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def compute_three_score(y_true, y_pred):
    y_pred = y_pred.cpu().numpy()
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            if y_pred[i][j] > 0.5:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    y_true = y_true.cpu().numpy()

    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')

    recall_micro = recall_score(
        y_true, y_pred, average='micro', zero_division=1)
    recall_macro = recall_score(
        y_true, y_pred, average='macro', zero_division=1)

    f1_score_micro = f1_score(
        y_true, y_pred, average='micro', zero_division=1)
    f1_score_macro = f1_score(
        y_true, y_pred, average='macro', zero_division=1)

    print('******************** mirco *************************')
    print('mirco_precision = ', precision_micro)
    print('mirco_recall    = ', recall_micro)
    print('mirco_F1        = ', f1_score_micro)
    print('******************** marco *************************')
    print('marco_Precision = ', precision_macro)
    print('marco_Recall    = ', recall_macro)
    print('marco_F1        = ', f1_score_macro)
    print('\n')
    recall = []
    prec = []
    for i in range(14):
        recall.append(recall_score(y_true[:, i], y_pred[:, i]))
        prec.append(precision_score(y_true[:, i], y_pred[:, i]))
    print('*' * 30)
    for i in range(len(opt.classes)):
        print('The AUROC of {} is {}'.format(opt.classes[i], recall[i]), prec[i])
    print('*' * 30)


def test(device='cuda'):
    # configure model
    basic_model = ResNet101(num_class=14).to(device)
    model = MyGCN(basic_model, 14).to(device)
    checkpoint = torch.load('your_checkpoint_name')
    model.load_state_dict(checkpoint['state_dict'])
    print('Done')

    # data
    test_data = ChestXrayDataSet(opt.data_root, opt.test_data_list)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / opt.batch_size)
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # test
    print('\n---------------------------------')
    print('...... Start testing ......')
    print('---------------------------------\n')

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, (data, label) in bar:
            inp = data.clone().detach().to(device)
            target = label.clone().detach().to(device)
            out1, out2, cc, _ = model(inp)
            output = (out1 + out2) / 2
            gt = torch.cat((gt, target), 0)
            pred = torch.cat((pred, output.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('*' * 30)
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    print('*' * 30)
    for i in range(len(opt.classes)):
        print('The AUROC of {} is {}'.format(opt.classes[i], AUROCs[i]))
    print('*' * 30)

    compute_three_score(gt, pred)
    return


def train(device='cuda'):
    basic_model = ResNet101(num_class=14).to(device)
    checkpoint = torch.load('your_checkpoint_name')
    basic_model.load_state_dict(checkpoint['state_dict'])

    model = MyGCN(basic_model, 14).to(device)

    # step2: data
    train_data = ChestXrayDataSet(opt.data_root, opt.train_data_list)
    val_data = ChestXrayDataSet(opt.data_root, opt.valid_data_list)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    parameter1 = list(map(id, model.gcn.fc.parameters()))
    parameter2 = filter(lambda p: id(p) not in parameter1, model.parameters())
    optimizer1 = torch.optim.Adam(model.gcn.fc.parameters(), lr=0.0005, betas=opt.betas,
                                 eps=opt.eps, weight_decay=opt.weight_decay)
    optimizer2 = torch.optim.Adam(parameter2, lr=0.0005, betas=opt.betas,
                                 eps=opt.eps, weight_decay=opt.weight_decay)

    CosineLR1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=200, eta_min=0.000001)
    CosineLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200, eta_min=0.000001)

    # step4: meters
    loss_mean_min = 1e100

    # train
    print('\n---------------------------------')
    print('...... Start training ......')
    print('---------------------------------\n')
    for epoch in range(5):
        print('Epoch', epoch + 1)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)

        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        for i, (data, label) in bar:
            model.train()
            torch.set_grad_enabled(True)
            inp = data.clone().detach().requires_grad_(True).to(device)
            target = label.clone().detach().to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            out1, out2, adj, h = model(inp)
            loss1 = model.dice_loss_input(adj, target)
            loss2 = model.classifier_loss(out1, out2, target, adj)
            loss = loss1+loss2
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            CosineLR1.step()
            CosineLR2.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        loss_mean = _val(model, val_dataloader, total_batch)
        time_end = time.strftime('%m%d_%H%M%S')
        if loss_mean_min > loss_mean:
            loss_mean_min = loss_mean
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer1.state_dict()},
                       'your_path' + time_end + '.pth.tar')
            print('Epoch [' + str(epoch + 1) + '] [save] [m_' + time_end + '] loss= ' + str(loss_mean))
        else:
            print('Epoch [' + str(epoch + 1) + '] [----] [m_' + time_end + '] loss= ' + str(loss_mean))
        print('--------------------------------------------------------------------------\n')
    return


def _val(model, dataloader, total_batch, device='cuda'):
    model.eval()
    counter = 0
    loss_sum = 0

    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, (data, label) in bar:
            inp = data.clone().detach().to(device)
            target = label.clone().detach().to(device)
            out1, out2, adj, h = model(inp)

            loss1 = model.dice_loss_input(adj, target)
            loss2 = model.classifier_loss(out1, out2, target, adj)
            loss = loss1 + loss2
            loss_sum += loss.item()
            counter += 1
            bar.set_postfix_str('loss: %.5s' % loss.item())

    loss_mean = loss_sum / counter
    return loss_mean


if __name__ == '__main__':
    print('--- CheXpert Baseline Classifier ---')
    mission = 2

    if mission == 1:
        train()
    elif mission == 2:
        test()


