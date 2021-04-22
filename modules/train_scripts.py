import torch
from tqdm import tqdm


def fit_epoch(model, train_loader, criterion, optimizer, scheduler, device='cuda'):
    """
    values for params by default
    :param model: efficient_net
    :param train_loader: data loader
    :param criterion: cross entropy
    :param optimizer: adam
    :param scheduler: lambdaLR
    :param device : {str}
    :return :{(float, float)}  tr_loss, tr_acc
    """

    current_loss = 0.0
    current_corrects = 0
    processed_data = 0
    batch_size = train_loader.batch_size
    it = 0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        it += 1
        scheduler.step()

        preds = torch.argmax(outputs, dim=1)
        current_corrects += torch.sum(preds == labels)

        processed_data += batch_size
        current_loss += loss.item() * batch_size
        if it % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'it: {it} lr: {lr} train_loss:{current_loss / processed_data} train_acc:{current_corrects.cpu().numpy() / processed_data}')


    train_loss = current_loss / processed_data 
    train_acc = current_corrects.cpu().numpy() / processed_data

    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion, device='cuda'):
    """
    values for params by default
    :param model: efficient_net
    :param val_loader: data loader
    :param criterion: cross entropy
    :param device: {str}
    :return :{(float, float)}  val_loss, val_acc
    """

    current_loss = 0.0
    current_corrects = 0
    processed_size = 0
    it = 0
    batch_size = val_loader.batch_size
    model.eval()
    for inputs, labels in val_loader:
        it += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        current_loss += loss.item() * batch_size
        current_corrects += torch.sum(preds == labels.data)
        processed_size += batch_size
        
    val_loss = current_loss / processed_size
    val_acc = current_corrects.cpu().numpy() / processed_size
    return val_loss, val_acc


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, start_epoch, writer, device, checkpoints_path):
    """
    Results of training and evaluation procedures are accumulated in this function
    as well as saving model and adding results to tensorboard
    :param model: efficient net
    :param train_loader: data loader
    :param val_loader: data loader
    :param optimizer: adam
    :param scheduler: lambdaLR
    :param criterion: cross entropy
    :param epochs: {int}
    :param start_epoch: {int}
    :param writer: tensorboard
    :param device: {str}
    :param checkpoints_path: {str}
    :return: None
    """

    least_val_loss = float('inf')

    for epoch in tqdm(range(start_epoch, epochs)):

        train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        writer.add_scalars("loss", {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalars("accuracy", {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

        print(f' \n epoch: {epoch} \n train_loss:{train_loss} \n val_loss:{val_loss}'
              f'\n train_acc:{train_acc}\n val_acc:{val_acc} \n')

        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_path + '/model.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoints_path + '/checkpoints.pth')

