import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
from absl import flags, app
from albumentations.augmentations.transforms import Resize,Normalize
from albumentations import Compose

try:
    from apex import amp, optimizers
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import torch.nn as nn
import torch.optim as optim
import lr_scheduler as set_lr

from dataset import load_train_data
import augmentations as aug
import models
import constants as cons
import time

flags.DEFINE_string('case', None, 'set case.', short_name='c')
flags.DEFINE_string('model_name', None, 'set model_name.', short_name='m')
flags.DEFINE_integer('final_epoch', 800, 'set final_epoch.', short_name='e')
flags.DEFINE_string('train_images_path', 'data', 'set train_images_path.', short_name='ti')
flags.DEFINE_string('train_labels_path', 'data', 'set train_labels_path.', short_name='tl')
flags.DEFINE_string('params_path', 'params', 'set params_path.', short_name='p')
flags.DEFINE_string('logs_path', 'logs', 'set logs_path.', short_name='l')
# cross validation
flags.DEFINE_integer('nfold', 0, 'set nfold.', short_name='nf')
# sampling
flags.DEFINE_bool('over_sampling',False, 'set over_sampling.', short_name='s')
# augmentation
flags.DEFINE_bool('augmentation', True, 'set augmentation.')
flags.DEFINE_bool('mixup', False, 'set mixup.', short_name='mix')
flags.DEFINE_bool('augmix', False, 'set augmix.')
flags.DEFINE_bool('aug_decrease', False, 'set aug_decrease.')
# restart
flags.DEFINE_integer('executed_epoch', 0, 'set executed_epoch.')
flags.DEFINE_bool('restart_from_final', False, 'set restart_from_final.')
flags.DEFINE_string('restart_param_path',None,'set restart_param_path.')
# loss influenced by with class weight
flags.DEFINE_bool('add_class_weight', False, 'set add_class_weight.', short_name='w')
# pseudo labeling
flags.DEFINE_float('confidence_border', None,'set confidence_border')
flags.DEFINE_string('result_path','result','set result_path.', short_name='r')
flags.DEFINE_string('result_case', None, 'set result_case.', short_name='c_r')
flags.DEFINE_string('test_images_path', 'data', 'set test_images_path.', short_name='test_i')

flags.DEFINE_integer('batch_size',64,'set batch_size')
flags.DEFINE_integer('num_worker',4,'set num_worker')
flags.DEFINE_string('opt_level','O1','set opt_level')

FLAGS = flags.FLAGS

def train_loop(model, loader, criterion, optimizer):
    '''Traning'''
    model.train()
    total_loss = 0
    for feed in tqdm(loader):
        # prepare Data
        inputs, labels = feed
        inputs, labels = inputs.cuda(), labels.cuda()
        # forward & calcurate Loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # initialize gradient
        optimizer.zero_grad()
        # backward
        with amp.scale_loss(loss,optimizer) as scaled_loss:#apex
            scaled_loss.backward()
        # update params
        optimizer.step()
        # totalize score
        total_loss += loss.item()
    return total_loss

def valid_loop(model, loader, criterion):
    '''Validation'''
    model.eval()
    with torch.no_grad():
        total_loss, total_correct, total_num = 0, 0, 0
        for feed in tqdm(loader):
            # prepare Data
            inputs, labels = feed
            label_size = labels.size(0)
            inputs, labels = inputs.cuda(), labels.cuda()
            # forward
            outputs = model(inputs)
            # calcurate Loss
            loss = criterion(outputs, labels)
            ## count correct answer
            pred = outputs.data.max(1, keepdim=True)[1]
            correct = pred.eq(labels.data.max(1,keepdim=True)[1]).sum()
            # totalize score
            total_loss += loss.item() * label_size
            total_correct += correct.item()
            total_num += label_size
    return total_loss / total_num, total_correct / total_num * 100

def main(argv=None):

    transform = Compose([
        Resize(cons.IMAGE_SIZE,cons.IMAGE_SIZE),
        Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)])
    valid_loader = load_train_data(
        train_images_path=FLAGS.train_images_path,
        train_labels_path=FLAGS.train_labels_path,
        batch_size=FLAGS.batch_size,
        num_worker=FLAGS.num_worker,
        valid=True,
        nfold=FLAGS.nfold,
        transform=transform)

    model = models.get_model(model_name=FLAGS.model_name, num_classes=cons.NUM_CLASSES)
    model.cuda()
    #model = torch.nn.DataParallel(model)
    
    DIR = '/'+ FLAGS.case +'/'+ FLAGS.model_name +'/fold'+ str(FLAGS.nfold)
    RESULT_PATH = ''
    if FLAGS.confidence_border is not None:
        DIR = DIR +'/with_pseudo_labeling'
        RESULT_PATH = RESULT_PATH + FLAGS.result_path
        if FLAGS.result_case is not None:
            RESULT_PATH = RESULT_PATH +'/'+ FLAGS.result_case        
        RESULT_PATH = RESULT_PATH + '/inference_with_c.csv'
        
        
    PARAM_DIR = FLAGS.params_path + DIR
    os.makedirs(PARAM_DIR,exist_ok=True)
    PARAM_NAME = PARAM_DIR + '/'+FLAGS.case
    if FLAGS.executed_epoch > 0:
        TRAINED_PARAM_PATH = FLAGS.restart_param_path +'/'+FLAGS.case+ str(FLAGS.executed_epoch)
        restart_epoch = FLAGS.executed_epoch + 1
        if FLAGS.restart_from_final:
            TRAINED_PARAM_PATH = TRAINED_PARAM_PATH + '_final'
        TRAINED_PARAM_PATH = TRAINED_PARAM_PATH + '.pth'
        model.load_state_dict(torch.load(TRAINED_PARAM_PATH))
    else:
        restart_epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=cons.start_lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level=FLAGS.opt_level)

    if FLAGS.add_class_weight:
        loader = load_train_data(
            train_images_path=FLAGS.train_images_path,
            train_labels_path=FLAGS.train_labels_path,
            batch_size=FLAGS.batch_size,
            num_worker=FLAGS.num_worker,
            nfold=FLAGS.nfold)
        count_label = np.zeros(10,dtype=np.int64)
        for feed in loader:
            _,labels = feed
            count_label += np.sum(labels.numpy().astype(np.int64),axis=0)
        weight=torch.from_numpy(count_label).cuda()
    else:
        weight=None
    criterion = nn.BCEWithLogitsLoss(weight=weight)
    
    writer = SummaryWriter(log_dir=FLAGS.logs_path + DIR + '/tensorboardX/')
    best_acc = 0


    if FLAGS.augmentation and FLAGS.aug_decrease:
        p = 0.5
        
        for e in range(restart_epoch,FLAGS.final_epoch):
            p_partical = p*(FLAGS.final_epoch-e)/FLAGS.final_epoch
            
            lr = set_lr.cosine_annealing(optimizer, cons.start_lr,e,100)
            writer.add_scalar('LearningRate', lr, e)
            
            train_loader = load_train_data(
                train_images_path=FLAGS.train_images_path,
                train_labels_path=FLAGS.train_labels_path,
                batch_size=FLAGS.batch_size,
                num_worker=FLAGS.num_worker,            
                nfold=FLAGS.nfold,
                confidence_border=FLAGS.confidence_border,
                result_path=RESULT_PATH,
                test_images_path=FLAGS.test_images_path,
                over_sampling=FLAGS.over_sampling,
                transform_aug=Compose([
                    aug.HueSaturationValue(p=p_partical),
                    aug.RandomBrightnessContrast(p=p_partical),
                    aug.CLAHE(p=p_partical),
                    aug.JpegCompression(p=p_partical),
                    aug.GaussNoise(p=p),
                    aug.MedianBlur(p=p),
                    aug.ElasticTransform(p=p_partical),
                    aug.HorizontalFlip(p=p),
                    aug.Rotate(p=p),
                    aug.CoarseDropout(p=p_partical),
                    aug.RandomSizedCrop(p=p)]),
                mixup=FLAGS.mixup,
                transform=transform)
            
            train_loss = train_loop(model, train_loader, criterion, optimizer)
            writer.add_scalar('train_loss', train_loss, e)
            
            valid_loss, valid_acc = valid_loop(model,valid_loader, criterion)
            writer.add_scalar('valid_loss', valid_loss, e)
            writer.add_scalar('valid_acc', valid_acc, e)

            print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy:{:.2f}'.format(e+1, train_loss, valid_loss, valid_acc))
            if e%10 == 0:
                torch.save(model.state_dict(), PARAM_NAME +'_'+ str(e)+'.pth')
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), PARAM_NAME+'_best.pth')
    else:
        
        if FLAGS.augmentation and not FLAGS.augmix:
            transform_aug = Compose([
                aug.HueSaturationValue(),
                aug.RandomBrightnessContrast(),
                aug.CLAHE(),
                aug.JpegCompression(),
                aug.GaussNoise(),
                aug.MedianBlur(),
                aug.ElasticTransform(),
                aug.HorizontalFlip(),
                aug.Rotate(),
                aug.CoarseDropout(),
                aug.RandomSizedCrop()])
        else:
            transform_aug=None

        train_loader = load_train_data(
            train_images_path=FLAGS.train_images_path,
            train_labels_path=FLAGS.train_labels_path,
            batch_size=FLAGS.batch_size,
            num_worker=FLAGS.num_worker,
            valid=False,
            nfold=FLAGS.nfold,
            over_sampling=FLAGS.over_sampling,
            transform_aug=transform_aug,
            augmix=FLAGS.augmix,
            mixup=FLAGS.mixup,
            transform=transform)
        
        total_time = 0
        for e in range(restart_epoch,FLAGS.final_epoch):
            start = time.time()
            lr = set_lr.cosine_annealing(optimizer, cons.start_lr,e,100)
            writer.add_scalar('LearningRate', lr, e)
            train_loss = train_loop(model, train_loader, criterion, optimizer)
            writer.add_scalar('train_loss', train_loss, e)
            valid_loss, valid_acc = valid_loop(model,valid_loader, criterion)
            writer.add_scalar('valid_loss', valid_loss, e)
            writer.add_scalar('valid_acc', valid_acc, e)
            print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy:{:.2f}'.format(e+1, train_loss, valid_loss, valid_acc))
            if e%10 == 0:
                torch.save(model.state_dict(), PARAM_NAME +'_'+ str(e)+'.pth')
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), PARAM_NAME+'_best.pth')
            total_time = total_time + (time.time()-start)
            print('average time: {}[sec]'.format(total_time/(e+1)))

    torch.save(model.state_dict(),PARAM_NAME +'_'+ str(FLAGS.final_epoch-1)+'_final.pth')


if __name__ == '__main__':
    flags.mark_flags_as_required(['case','model_name'])
    app.run(main)
