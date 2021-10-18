'''Train MNIST with PyTorch.'''
import os
import argparse

from torchsummary import summary
import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn

from models import ResNet18
from utils import progress_bar

print(f"torch.__version__(): {torch.__version__}")

print(f"torchvision.__version__(): {torchvision.__version__}")

if torch.cuda.is_available():
    print("Using GPU...")
    DEVICE = 'cuda'
else:
    print("Using CPU...")
    DEVICE = 'cpu'

# pylint: disable=R0914,R0915
def run():
    """Run the main loop as per pytorch requirements
    """
    # torch.multiprocessing.freeze_support()
    print('loop')

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
    # print(f"np.shape(trainset): {np.shape(trainset)}")
    print(f"type(trainset): {type(trainset)}")
    print(f"trainset.__len__(): {trainset.__len__()}")
    print(f"trainset.__sizeof__(): {trainset.__sizeof__()}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #         'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18(1)
    net = net.to(DEVICE)
    if DEVICE == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    summary(net, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)


    # Training
    def train(epoch):
        print(f"\nEpoch: {epoch}")
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                len(trainloader),
                f"""Loss: {train_loss/(batch_idx+1):%.3f}
                | Acc: {100.*correct/total}%% ({correct/total})""")

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        best_acc = 0  # best test accuracy
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx,
                    len(trainloader),
                    f"""Loss: {test_loss/(batch_idx+1):%.3f}
                    | Acc: {100.*correct/total}%% ({correct/total})""")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)

if __name__ == '__main__':
    run()
