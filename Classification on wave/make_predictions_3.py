
from util import *
from data_loader import *
from network import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def predict_one_model_with_wave(checkpoint, frame):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    # model = checkpoint['model']
    model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))
    
    win_size = config.audio_length
    stride = int(config.sampling_rate * 0.2)

    top1 = AverageMeter()
    top3 = AverageMeter()

    start = time.time()

    if config.cuda is True:
        model.cuda()
    model.eval()

    file_names = []
    prediction = torch.zeros((1, 16)).cuda()
#     label_all=[]
    
    with torch.no_grad():

        for idx in tqdm(range(frame.shape[0])):
            filename = os.path.splitext(frame["fname"][idx])[0] + '.pkl'
            file_path = os.path.join(config.data_dir, filename)
            record_data = load_data(file_path)
            label_idx = frame["label_idx"][idx]

            if len(record_data) < win_size:
                record_data = np.pad(record_data, (0, win_size - len(record_data)), "constant")

            wins_data = []
            for j in range(0, len(record_data) - win_size + 1, stride):
                win_data = record_data[j: j + win_size]

                maxamp = np.max(np.abs(win_data))
                if maxamp < 0.005 and j > 1:
                    continue
                wins_data.append(win_data)

            # print(file_path, len(record_data)/config.sampling_rate, len(wins_data))

            if len(wins_data) == 0:
                print(file_path)

            wins_data = np.array(wins_data)

            wins_data = wins_data[:, np.newaxis, :]

            data = torch.from_numpy(wins_data).type(torch.FloatTensor)
            
            label = torch.FloatTensor([label_idx])

            if config.cuda:
                data, label = data.cuda(), label.cuda()

            output = model(data)
            output = torch.sum(output, dim=0, keepdim=True)
            prec1, prec3 = accuracy(output, label, topk=(1, 3))
            top1.update(prec1)
            top3.update(prec3)

            prediction = torch.cat((prediction, output), dim=0)
            file_names.append(frame["fname"][idx])
#             label_all.append(label)

        elapse = time.strftime('%Mm:%Ss', time.gmtime(time.time() - start))
        logging.info(' Test on file: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Time: {elapse}'
              .format(top1=top1, top3=top3, elapse=elapse))
    print((' Test on file: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Time: {elapse}'
              .format(top1=top1, top3=top3, elapse=elapse)))

#             file_names.append(frame["fname"][idx])

    prediction = prediction[1:]
    

    return file_names, prediction


def model_ensembles(valid_path='./whale_valid.csv',mean_method='arithmetic'):
    """
    make two prediction files for stacking. One for train and one for test. 
    Prediction matrix of samples x classes (9400 x 41)
    
    :param mean_method: 'arithmetic' or 'geometric'
    """

    model_dir = config.model_dir

    # make test prediction
    valid = pd.read_csv(valid_path, sep='\t')
    LABELS = list(valid.label.unique())
    
    label_idx = {label: i for i, label in enumerate(LABELS)}
    valid.set_index("fname")
    idxes = valid.label.apply(lambda x: label_idx[x]).values
    y_all=[]
    
    for li in idxes:
        y=np.zeros(len(LABELS))
        y[li]=1
        y_all.append(y)
    valid["label_idx"] = y_all
    label_all = np.array(y_all)
    frame = valid

    pred_list = []

    for i in range(config.n_folds):
#     for i in range(1):
        ckp = config.model_dir + '/model_best.'+ str(i) + '.pth.tar'
        fn, pred = predict_one_model_with_wave(ckp, frame)

        # pred = pred.cpu().numpy()
        pred_list.append(pred)

    if mean_method == 'arithmetic':
        predictions = torch.zeros_like(pred_list[0]).cuda()
        for pred in pred_list:
            predictions = predictions + pred
        predictions = predictions / len(pred_list)

    elif mean_method == 'geometric':
        predictions = torch.ones_like(pred_list[0]).cuda()
        for pred in pred_list:
            predictions = predictions * pred
        predictions = predictions ** (1. / len(pred_list))
    else:
        print('mean_method not specified.')
        
    prec1, prec3 = accuracy(predictions, label_all, topk=(1, 3))

    pred_labels = F.softmax(predictions, dim=1)

    save_to_csv(fn, pred_labels.cpu().numpy(), 'Ensemble_predictions.csv')
    logging.info(' Accuracy on file based on ensembled models: Prec@1 {top1:.3f} Prec@3 {top3:.3f}'
              .format(top1=prec1, top3=prec3))
    
    print(' Accuracy on file based on ensembled models: Prec@1 {top1:.3f} Prec@3 {top3:.3f}'
              .format(top1=prec1, top3=prec3))
    print('Estimation Completed!')
    
def save_to_csv(files_name, prediction, file):
    df = pd.DataFrame(index=files_name, data=prediction)
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path, header=None)


def test():
    a = ["1","2","3"]
    b = ["1","2","3"]
    print(a!=b)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    config = Config(debug=False,
                sampling_rate=22050,
                audio_duration=1.5,
                data_dir='./pickle_files',
                arch='waveResnext101_32x4d',
                lr=0.01,
                pretrain='imagenet',
                model_dir='./model',
                prediction_dir='./prediction',
                batch_size=24,
                epochs=50)

    # test()

    model_ensembles(valid_path='./whale_valid.csv', mean_method='arithmetic')

    
    
    