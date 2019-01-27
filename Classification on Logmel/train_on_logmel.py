from core import *
from data_loader import *
from util import *

import torch.optim as optim
import torch.backends.cudnn as cudnn
import time

def main():

    train = pd.read_csv('./whale_train.csv', sep='\t')

    LABELS = list(train.label.unique())
    
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")
    idxes = train.label.apply(lambda x: label_idx[x]).values
    y_all=[]
    
    for li in idxes:
        y=np.zeros(len(LABELS))
        y[li]=1
        y_all.append(y)
    train["label_idx"] = y_all
#     split_holder = train.label.apply(lambda x: label_idx[x])

    if config.debug == True:
        train = train[:500]

#     skf = StratifiedKFold(n_splits=config.n_folds)
    skf = StratifiedKFold(n_splits=config.n_folds, random_state=None, shuffle=True)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train, idxes)):

        end = time.time()
        # split the dataset for cross-validation
        train_set = train.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)


        logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
              .format(foldNum, len(train_set), len(val_set)))

        # define train loader and val loader
        trainSet = Freesound_logmel(config=config, frame=train_set, transform=transforms.Compose([ToTensor()]), mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = Freesound_logmel(config=config, frame=val_set, transform=transforms.Compose([ToTensor()]), mode="train")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = run_method_by_string(config.arch)(pretrained=config.pretrain)

        if config.cuda:
            model.cuda()

        # define loss function (criterion) and optimizer
#         criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.BCEWithLogitsLoss().cuda()
        
#         train_criterion = F.binary_cross_entropy_with_logits().cuda()
#         val_criterion = F.binary_cross_entropy_with_logits().cuda()

        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=config.lr)

        cudnn.benchmark = True

        train_on_fold(model, criterion, criterion, optimizer, train_loader, val_loader, config, foldNum)
        
        #estimate the accuracy on wave files instead of on wave segments
#         val_on_file_wave(model, config, val_set)
# 
#         time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
#         logging.info("--------------Time on fold {}: {}--------------\n"
#               .format(foldNum, time_on_fold))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    config = Config(debug=False,
                    sampling_rate=22050,
                    audio_duration=1.5,
                    data_dir='./pickle_files',
                    arch='resnext101_32x4d_',
                    lr=0.01,
                    pretrain='imagenet',
                    model_dir='./model',
                    prediction_dir='./prediction',
                    batch_size=24,
                    epochs=40,
                    n_mels=128, 
                    frame_weigth=128, 
                    frame_shift=10)

#     if os.path.exists(config.model_dir):
#         shutil.rmtree(config.model_dir)
#     os.makedirs(config.model_dir)
    # create log
    logging = create_logging('./log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)
    main()
    
    