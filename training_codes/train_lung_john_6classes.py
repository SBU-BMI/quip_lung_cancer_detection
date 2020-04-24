import argparse
from torchvision import transforms
import time, os, sys
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score, confusion_matrix
import copy
from torch.utils.data import DataLoader, Dataset
import pdb
from utils import *
import glob
import collections
import random

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='', type=str, help='model')
parser.add_argument('--net_depth', default=34, type=int)
parser.add_argument('--weighted', type=str2bool, default=False, help="apply weight to the loss")
parser.add_argument('--sampler', type=str2bool, default=False, help="using sampler method")
parser.add_argument('--acinar_ratio', type=float, default=1, help="the ratio of acinar patches in the training data")
parser.add_argument('--APS', default = 350, type = int)
parser.add_argument('--clean_up', default=0.25, type=float, help='percentage of training samples to be ignored from Lapidic and Micropapillary')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_benign', default=80_000, type=int)
parser.add_argument('--random_seed', default=2954321, type=int)
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs in training')
parser.add_argument('--check_after', default=2, type=int, help='check the network after check_after epoch')
parser.add_argument('--note', type=str, default=None, help="note while running the code")

args = parser.parse_args()
if args.random_seed is not None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

args.net_type = '{}_netDepth-{}_APS-{}_randomSeed-{}_numBenign-{}'.format(os.path.basename(__file__).split('.')[0],\
                            str(args.net_depth), str(args.APS), str(args.random_seed), str(args.num_benign))

if args.note is not None:
    args.net_type += '_note-' + args.note

with open(os.path.basename(__file__)) as f:
    codes = f.readlines()
print('\n\n' + '=' * 20 + os.path.basename(__file__) + '=' * 20)
for c in codes:
    print(c[:-1])

with open('prostate_utils.py') as f:
    codes = f.readlines()
print('\n\n' + '=' * 20 + 'prostate_utils.py' + '=' * 20)
for c in codes:
    print(c[:-1])

print(args)
use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)
device = torch.device("cuda:0")

mean = [0.6462,  0.5070,  0.8055]      # for Prostate cancer
std = [0.1381,  0.1674,  0.1358]

mean =  [0.8301,  0.6600,  0.8054]   # for seer lung john
std = [ 0.0864,  0.1602,  0.0647]

APS = args.APS      # default = 448
input_size = 224
n_class = 6

data_transforms = {
    'train': transforms.Compose([           # 2 steps of data augmentation for training
        transforms.RandomCrop(APS),       # perform random crop manually in the dataloader
        transforms.Scale(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

    'val': transforms.Compose([
        transforms.CenterCrop(APS),
        transforms.Scale(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
}

lepidic_fol_1 = '/data10/shared/hanle/extract_lung_Rutger_seer_john/clean_up_each_5000/sample_lepidic_5000'
lepidic_fol_2 = '/data10/shared/hanle/extract_lung_Rutger_seer_john/clean_up_each_5000/sample_lepidic_6600'
acinar_fol_1 = '/data10/shared/hanle/extract_lung_Rutger_seer_john/clean_up_each_5000/sample_acinar_5000'
acinar_fol_2 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_acinar_solid/patches_lung_seer_john_additional_acinar_web'
micropap_fol_1 = '/data10/shared/hanle/extract_lung_Rutger_seer_john/clean_up_each_5000/sample_micropap_5000'
#micropap_fol_2 = '/data10/shared/hanle/extract_lung_Rutger_seer_john/clean_up_each_5000/sample_micropap_additional_6000'
micropap_fol_3 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap/patches_lung_seer_john_additional_micropap_web'
mucinous_fol = '/data10/shared/hanle/extract_lung_Rutger_seer_john_mucinous/patches_lung_seer_john_mucinous_web'
solid_fol = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_acinar_solid/patches_lung_seer_john_solid_web'

lepidic_fol_3 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap_020220/patches_lung_seer_john_020220/patches_lung_seer_john_020220_lepidic_web'
acinar_fol_3 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap_020220/patches_lung_seer_john_020220/patches_lung_seer_john_020220_acinar_web'
micropap_fol_4 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap_020220/patches_lung_seer_john_020220/patches_lung_seer_john_020220_micropap_web'
mucinous_fol_2 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap_020220/patches_lung_seer_john_020220/patches_lung_seer_john_020220_mucinous_web'
solid_fol_2 = '/data10/shared/hanle/extract_lung_Rutger_seer_john_additional_micropap_020220/patches_lung_seer_john_020220/patches_lung_seer_john_020220_solid_web'

condition = lambda f:f[-5] in {'0', '2', '3', '5', '6'}
lepidic_fns = [f for f in glob.glob(lepidic_fol_1 + '/*.png') + glob.glob(lepidic_fol_2 + '/*.png') + glob.glob(lepidic_fol_3 + '/*.png') if condition(f)]
acinar_fns = [f for f in glob.glob(acinar_fol_1 + '/*.png') + glob.glob(acinar_fol_2 + '/*.png') + glob.glob(acinar_fol_3 + '/*.png') if condition(f)]
micropap_fns = [f for f in glob.glob(micropap_fol_1 + '/*.png') + glob.glob(micropap_fol_3 + '/*.png') + glob.glob(micropap_fol_4 + '/*.png')if condition(f)]
mucinous_fns = [f for f in glob.glob(mucinous_fol + '/*.png') + glob.glob(mucinous_fol_2 + '/*.png') if condition(f)]
solid_fns = [f for f in glob.glob(solid_fol + '/*.png') +  glob.glob(solid_fol_2 + '/*.png') if condition(f)]


print('lepidic fns: ', len(lepidic_fns))
print('acinar fns: ', len(acinar_fns))
print('micropap fns: ', len(micropap_fns))
print('mucinous fns: ', len(mucinous_fns))
print('solid fns: ', len(solid_fns))

def find_slides_from_fns(fns):
    slides = [f.split('/')[-1].split('.')[0] for f in fns]
    counters = collections.Counter(slides)
    out = []
    for slide, cnt in counters.items():
        if 300 > cnt > 15:
            out.append(slide)
    random.shuffle(out)
    return out

lepidic_slides = find_slides_from_fns(lepidic_fns)
acinar_slides = find_slides_from_fns(acinar_fns)
micropap_slides = find_slides_from_fns(micropap_fns)
mucinous_slides = find_slides_from_fns(mucinous_fns)
solid_slides = find_slides_from_fns(solid_fns)

acinar_slides = [s for s in acinar_slides if s not in lepidic_slides + micropap_slides + mucinous_slides + solid_slides]

print('lepidic slides: ', len(lepidic_slides))
print('acinar slides: ', len(acinar_slides))
print('micropap slides: ', len(micropap_slides))
print('mucinous slides: ', len(mucinous_slides))
print('solid slides: ', len(solid_slides))

val_slides = lepidic_slides[:5] + acinar_slides[:10] + micropap_slides[:5] + mucinous_slides[:4] + solid_slides[:4]
print('val slides: ', val_slides, len(val_slides))

benign_fns = [f for f in glob.glob('/data2/patches_lung_seer_john/*1.png') +\
                glob.glob('/data2/patches_lung_seer_john_additional/*1.png') +\
                glob.glob('/data2/patches_lung_seer_john_validation/*1.png')]
random.shuffle(benign_fns)
print('number of benign patches: ', len(benign_fns))
benign_fns = benign_fns[:args.num_benign]
benign_fns_val = [f for f in benign_fns if f.split('/')[-1].split('.')[0] in val_slides]
random.shuffle(benign_fns_val)
benign_fns_val = benign_fns_val[:1_000]
benign_fns_train = [f for f in benign_fns if f.split('/')[-1].split('.')[0] not in val_slides]

all_fns = lepidic_fns + acinar_fns + micropap_fns + mucinous_fns + solid_fns
img_vals = [f for f in all_fns if f.split('/')[-1].split('.')[0] in val_slides] + benign_fns_val
img_trains = [f for f in all_fns if f.split('/')[-1].split('.')[0] not in val_slides] + benign_fns_train

print('len of train/val set: ', len(img_trains), len(img_vals))

stats = collections.defaultdict(int)
for fn in img_trains:
    stats[fn[-5]] += 1
print('training data stats: ', stats)

stats = collections.defaultdict(int)
for fn in img_vals:
    stats[fn[-5]] += 1
print('val data stats: ', stats)

train_set = data_loader(img_trains, transform = data_transforms['train'])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

val_set = data_loader(img_vals, transform = data_transforms['val'])
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def val_fn_epoch(val_fn = None, crit = None, val_loader = None):
    nline = 0
    running_loss = 0.0
    labels_val = torch.zeros(0).type(torch.LongTensor)
    preds_val = torch.zeros(0).type(torch.LongTensor).to(device)
    with torch.no_grad():
        for ix, batch in enumerate(val_loader):
            if (len(val_loader.dataset) - nline) < 2: continue
            inputs, targets = batch

            labels_val = torch.cat((labels_val, targets.type(torch.LongTensor)))
            inputs = Variable(inputs.to(device))
            targets = Variable(targets.type(torch.LongTensor).to(device))
            output = val_fn(inputs)
            if type(output) == tuple:
                output,_ = output
            N = output.size(0)

            loss = crit(output, targets)
            running_loss += loss.item() * N
            _, preds = torch.max(output.data, 1)        # get the argmax index along the axis 1
            preds_val = torch.cat((preds_val, preds))

    labels_val = labels_val.to(device)
    val_acc = accuracy_score(labels_val, preds_val)
    f1 = f1_score(labels_val, preds_val, average='macro')    # Calculate metrics for each label, and find their average weighted
    print(f1_score(labels_val, preds_val, average=None))      # print F1-score for each class

    unique, counts = np.unique(np.array(labels_val), return_counts=True)
    return val_acc, f1, preds_val, labels_val, running_loss/labels_val.size(0), dict(zip(unique, counts))


def train_model(model, criterion = None, num_epochs=100, train_loader = train_loader, val_loader = val_loader):
    best_f1 = 0
    best_epoch = 0
    start_training = time.time()

    for epoch in range(num_epochs):
        start = time.time()

        if epoch < 8: lr = args.lr
        elif epoch < 12: lr = args.lr/2
        elif epoch < 20: lr = args.lr/10
        elif epoch < 30: lr = args.lr / 50
        else: lr = args.lr/100

        if epoch >= 2:
            for param in model.parameters():
                param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('lr: {:.6f}'.format(lr))
        print('-' * 50)

        for phase in ['train']:
            if phase == 'train':
                data_loader = train_loader
                model.train(True)
            else:
                data_loader = val_loader
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            N_tot = 0
            labels_train = torch.zeros(0).type(torch.LongTensor)
            preds_train = torch.zeros(0).type(torch.LongTensor).to(device)
            for ix, data in enumerate(data_loader):
                if (len(data_loader.dataset) - N_tot) < 3: continue
                inputs, labels = data
                labels_train = torch.cat((labels_train, labels.type(torch.LongTensor)))

                inputs = Variable(inputs.to(device))
                labels = Variable(labels.type(torch.LongTensor).to(device))

                optimizer.zero_grad()
                outputs = model(inputs)
                if type(outputs) == tuple:  # for inception_v3 output
                    outputs,_ = outputs

                _, preds = torch.max(outputs.data, 1)   # preds are the index of the maximum element

                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                N_tot += outputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                preds_train = torch.cat((preds_train, preds))

            unique, counts = np.unique(np.array(labels_train), return_counts=True)
            print('| Epoch:[{}][{}/{}]\tTrain_Loss: {:.4f}\tAccuracy: {:.4f}\tTrain_data: {}\tTime: {:.2f} mins'.format(epoch + 1, ix + 1,
                 len(data_loader.dataset)//args.batch_size,
                 running_loss / N_tot, running_corrects.item() / N_tot, dict(zip(unique, counts)), (time.time() - start)/60.0))

            try:
                conf_matrix = confusion_matrix(labels_train.to(device), preds_train, labels=[i for i in range(n_class)])
                print(f1_score(labels_train.to(device), preds_train, average=None))
                print(conf_matrix)
            except:
                print('could not compute confusion matrix.')
            sys.stdout.flush()

            ############ VALIDATION #############################################
            if (epoch + 1) % args.check_after == 0:
                model.eval()
                start = time.time()
                val_acc, f1, Pr, Tr, val_loss, labels_val = val_fn_epoch(val_fn = model, crit = criterion, val_loader = val_loader)
                print("Epoch: {}\tVal_Loss: {:.4f}\tAccuracy: {:.4f}\tF1-score: {:.4f}\tVal_data: {}\tTime: {:.3f}mins".format(
                    (epoch + 1), val_loss, val_acc, f1, labels_val, (time.time() - start)/60.0))

                try:
                    conf_matrix = confusion_matrix(Tr, Pr, labels=[i for i in range(n_class)])
                    print(conf_matrix)
                except:
                    print('could not compute confusion matrix.')

                start = time.time()


                save_point = './checkpoint/350_589321/'
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                saved_model_fn = args.net_type + '_' + strftime('%m%d_%H%M')
                best_model = copy.deepcopy(model)
                state = {
                    'model': best_model,
                    'f1-score': f1,
                    'args': args,
                    'lr': lr,
                    'saved_epoch': epoch,
                }

                # deep copy the model
                if f1 > best_f1 and epoch > 2:
                    print('Saving model')
                    best_f1 = f1
                    best_epoch = epoch + 1
                    torch.save(state, save_point + saved_model_fn + '_bestF1_' + str(f1) + '_' + str(epoch) + '.t7')
                    print('=======================================================================')
                #else:
                #    torch.save(state, save_point + saved_model_fn + '_F1_' + str(f1) + '_' + str(epoch) + '.t7')


    time_elapsed = time.time() - start_training
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best F1-score: {:4f} at epoch: {}'.format(best_f1, best_epoch))



def main():
    sys.setrecursionlimit(10000)

    if args.net_depth == 34:
        model = models.resnet34(pretrained=True)
    elif args.net_depth == 50:
        model = models.resnet50(pretrained=True)
    elif args.net_depth == 101:
        model = models.resnet101(pretrained=True)
    elif args.net_depth == 152:
        model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, n_class)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    cudnn.benchmark = True
    print(model)
    print('Start training ... ')

    criterion = nn.CrossEntropyLoss().to(device)
    train_model(model, criterion, num_epochs=args.num_epochs, train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
    main()
