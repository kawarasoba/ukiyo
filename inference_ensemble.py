import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# makedir
import os
# augument
from absl import flags, app
# augmentation
from albumentations.augmentations import transforms
from albumentations.augmentations.transforms import Resize,Normalize
from albumentations import Compose

from dataset import load_test_data
from dataset import UkiyoeTestDataset
import models
import constants as cons
from convenient_function import fix_model_state_dict

flags.DEFINE_string('case', None, 'set case.', short_name='c')
flags.DEFINE_string('test_images_path', 'data', 'set train_images_path.', short_name='ti')
flags.DEFINE_string('params_path', 'params', 'set params_path.', short_name='p')
flags.DEFINE_string('result_path', 'result', 'set result_path.', short_name='r')
flags.DEFINE_bool('adjust_inference',False, 'set adjust_inference.')
# pseudo labeling
flags.DEFINE_bool('output_confidence' ,False, 'set output_confidence', short_name='confi')
FLAGS = flags.FLAGS

PARAM_LIST = [
    {'model':'/se_oversampling/senet154/fold0/with_pseudo_labeling/se_oversampling_best.pth','fix':False,'ratio':1},
    {'model':'/dense_oversampling/densenet161/fold0/with_pseudo_labeling/dense_oversampling_best.pth','fix':False,'ratio':2},
    {'model':'/seres_oversampling/se_resnet50/fold0/with_pseudo_labeling/seres_oversampling_best.pth','fix':False,'ratio':2},
    {'model':'/model_dense/densenet161/fold0/model_dense_880.pth','fix':False,'ratio':1},
    {'model':'/aug_decrease_effi_b3/efficientnet_b3/fold0/aug_decrease_effi_b3_790.pth','fix':True,'ratio':1},
    {'model':'/model_effi_b3/efficientnet_b3/fold0/model_effi_b3_980.pth','fix':True,'ratio':1},
    {'model':'/model_effi_b3_sampling/efficientnet_b3/fold0/with_pseudo_labeling/model_effi_b3_sampling_best.pth','fix':False,'ratio':2},
    #{'model':'/model_senet154/senet154/fold0/model_senet154_680.pth','fix':True,'ratio':1},
    {'model':'/partical_augmentation/senet154/fold0/partical_augmentation_380.pth','fix':True,'ratio':1},
    {'model':'/se_oversampling/se_resnet50/fold0/se_oversampling_490.pth','fix':False,'ratio':2},
    {'model':'/ince3_w/inceptionv3/fold0/ince3_weight_670.pth','fix':False,'ratio':1}
]

def inference_loop(model, loader,data_len):
    model.eval()
    output = np.zeros((data_len,cons.NUM_CLASSES))
    for idx, feed in enumerate(tqdm(loader)):
        _, inputs = feed
        inputs = inputs.cuda()
        #forward
        outputs = torch.sigmoid(model(inputs))
        output[idx] = outputs.data.cpu().numpy()
    return output

def adjust_inference(pred,confidence,data_len):
    inference_ratio = np.sum(np.identity(cons.NUM_CLASSES)[pred],axis=0)
    diff = np.round(inference_ratio - cons.adjust_ratio*data_len).astype(np.int32)
    print(diff)
    for excess_label in np.where(diff > 0)[0][::-1]:
        target_all_id = np.where(pred==excess_label)[0]
        target_id = target_all_id[
            np.argsort(confidence[target_all_id][:, excess_label])[:diff[excess_label]]]
        for idx in target_id:
            if 1 in np.sign(diff):
                label_candidates = np.argsort(confidence[idx])[::-1]
                print(np.sort(confidence[idx])[::-1])
                print(label_candidates)
                for label_candidate in label_candidates:
                    if diff[label_candidate] < 0:
                        pred[idx] = label_candidate
                        diff[excess_label]-=1
                        diff[label_candidate]+=1
                        print(excess_label,'==>',label_candidate)
                        print(diff)
                        break
    return pred

def main(argv=None):

    loader = load_test_data(
        data_path=FLAGS.test_images_path,
        batch_size=1,
        transform=Compose([
            Resize(cons.IMAGE_SIZE,cons.IMAGE_SIZE),
            Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)]))
    data_len = UkiyoeTestDataset(data_path=FLAGS.test_images_path).__len__()

    confidence_sum = np.zeros((data_len,cons.NUM_CLASSES))
    ratio_sum = 0
    for e in range(len(PARAM_LIST)):
        ratio_sum = ratio_sum + PARAM_LIST[e]['ratio']
        PARAM_PATH = FLAGS.params_path + PARAM_LIST[e]['model']
        model = models.get_model(model_name=PARAM_LIST[e]['model'].split('/')[2],num_classes=cons.NUM_CLASSES)
        if PARAM_LIST[e]['fix']:
            model.load_state_dict(fix_model_state_dict(torch.load(PARAM_PATH)))
        else:
            model.load_state_dict(torch.load(PARAM_PATH))
        confidence_sum = confidence_sum + inference_loop(model,loader,data_len)*PARAM_LIST[e]['ratio']
    pred = np.argmax(confidence_sum, axis=1)
    confidence = confidence_sum/ratio_sum
    if FLAGS.adjust_inference:
        pred = adjust_inference(pred,confidence,data_len)

    result = pd.DataFrame(columns=['id','y'])
    if FLAGS.output_confidence:
        result_with_c = pd.DataFrame(columns=['id','y','confidence'])
    
    for idx,feed in enumerate(loader):
        id,_ = feed
        result = result.append(
            pd.Series(
                [id.data.numpy()[0], pred[idx]],
                index=result.columns),
            ignore_index=True)
        if FLAGS.output_confidence:
            result_with_c = result_with_c.append(
                pd.Series(
                    [id.data.numpy()[0], pred[idx], confidence[idx,pred[idx]]],
                    index=result_with_c.columns),
                ignore_index=True)

    RESULT_DIR = FLAGS.result_path +'/'+ FLAGS.case
    os.makedirs(RESULT_DIR,exist_ok=True)
    result.to_csv(RESULT_DIR +'/inference.csv', index=False)
    if FLAGS.output_confidence:
        result_with_c[['id','y']] = result_with_c[['id','y']].astype(int)
        result_with_c.to_csv(RESULT_DIR +'/inference_with_c.csv', index=False)    

if __name__ == '__main__':
    flags.mark_flags_as_required(['case'])
    app.run(main)