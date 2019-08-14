import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
from common import *

# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg---------------
model_S = DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2, num_classes=num_classes).to(device)
# --------------Loss---------------------------
criterion_S = nn.CrossEntropyLoss().cuda()
# setup optimizer
optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)
# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset("./data_train", mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True)
    mri_data_val = H5Dataset("./data_val", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
    print('Rate     | epoch  | Loss seg| DSC_val')
    for epoch in range (num_epoch):
        scheduler_S.step(epoch)
        # zero the parameter gradients
        model_S.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            # Set mode cuda if it is enable, otherwise mode CPU
            images = images.to(device)
            targets = targets.to(device)           
            optimizer_S.zero_grad()
            outputs = model_S(images)
            loss_seg = criterion_S(outputs, targets) #Crossentropy loss for Seg
            loss_seg.backward()
            optimizer_S.step()

        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            for data_val in valloader:
                images_val, targets_val = data_val
                model_S.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)

                outputs_val = model_S(images_val)
                _, predicted = torch.max(outputs_val.data, 1)
                # ----------Compute dice-----------
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()
                dsc = []
                for i in range(1, num_classes):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)

                # outputs_val = model_S(images_val)
                # _, predicted = torch.max(outputs_val.data, 1)
                # # ----------Compute dice-----------
                # predicted = predicted.squeeze()
                # targets_val = targets_val.data[0].cpu().numpy()
                # dsc = []
                # for i in range(1, num_classes):  # ignore Background 0
                #     if (np.sum(targets_val[targets_val==i])>0):
                #         dsc_i = dice(predicted, targets_val, i)
                #         dsc.append(dsc_i)
                # dsc = np.mean(dsc)

        #-------------------Debug-------------------------
        for param_group in optimizer_S.param_groups:
            print('%0.6f | %6d | %0.5f | %0.5f ' % (\
                    param_group['lr'], epoch,
                    # loss_seg,
                    loss_seg.data.cpu().numpy(),
                    #dsc for center path
                    dsc))

        #Save checkpoint
        if (epoch % step_size_S) == 0 or epoch == (num_epoch - 1) or (epoch % 1000) == 0:
            torch.save(model_S.state_dict(), './checkpoints/' + '%s_%s.pth' % (str(epoch).zfill(5), checkpoint_name))

