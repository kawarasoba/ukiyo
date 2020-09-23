import torch
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
import models
import constants as cons
from convenient_function import fix_model_state_dict

flags.DEFINE_string('case', None, 'set case.', short_name='c')
flags.DEFINE_string('model_name', None, 'set model_name.', short_name='m')
flags.DEFINE_integer('executed_epoch', None, 'set executed_epoch.', short_name='e')
flags.DEFINE_integer('nfold', 0, 'set nfold.', short_name='nf')
flags.DEFINE_bool('is_best_param',False,'set is_best_param.',short_name='best')
flags.DEFINE_bool('is_final_epoch',False,'set is_final_epoch.',short_name='fe')
flags.DEFINE_bool('fix_state_dict',False,'set fix_state_dict.',short_name='fix')
flags.DEFINE_string('test_images_path', 'data', 'set train_images_path.', short_name='ti')
flags.DEFINE_string('params_path', 'params', 'set params_path.', short_name='p')
flags.DEFINE_string('result_path', 'result', 'set result_path.', short_name='r')
# pseudo labeling
flags.DEFINE_bool('output_confidence' ,False, 'set output_confidence', short_name='prob')
flags.DEFINE_bool('pseudo_labeling',False,'set pseudo_labeling.',short_name='pseudo')

FLAGS = flags.FLAGS

def inference_loop(model, loader,output_confidence):
    model.eval()
    if output_confidence:
        result = pd.DataFrame(columns=['id','y','confidence'])        
    else:
        result = pd.DataFrame(columns=['id','y'])

    for feed in tqdm(loader):
        id, inputs = feed
        inputs = inputs.cuda()
        #forward
        outputs = torch.sigmoid(model(inputs))
        pred = outputs.data.max(1)
        if output_confidence:
            result = result.append(
                pd.Series(
                    [id.data.numpy()[0], pred[1].data.cpu().numpy()[0], pred[0].data.cpu().numpy()[0]],
                    index=result.columns),
                ignore_index=True)
        else:
            result = result.append(
                pd.Series(
                    [id.data.numpy()[0], pred[1].data.cpu().numpy()[0]],
                    index=result.columns),
                ignore_index=True)
    result[['id','y']] = result[['id','y']].astype(int)
    return result

def main(argv=None):

    PARAM_PATH = FLAGS.params_path+'/'+FLAGS.case+'/'+FLAGS.model_name+'/fold'+str(FLAGS.nfold)
    RESULT_DIR = FLAGS.result_path+'/'+FLAGS.case+'/'+FLAGS.model_name
    if FLAGS.pseudo_labeling:
        PARAM_PATH = PARAM_PATH+'/with_pseudo_labeling'
        RESULT_DIR = RESULT_DIR+'/with_pseudo_labeling'        
    PARAM_PATH = PARAM_PATH+'/'+FLAGS.case
    RESULT_DIR = RESULT_DIR+'/'+FLAGS.case
    if FLAGS.is_best_param:
        PARAM_PATH = PARAM_PATH+'_best'
        RESULT_DIR = RESULT_DIR+'_best'
    else:
        PARAM_PATH = PARAM_PATH+'_' + str(FLAGS.executed_epoch)
        RESULT_DIR = RESULT_DIR+'_'+ str(FLAGS.executed_epoch)
        if FLAGS.is_final_epoch:
            PARAM_PATH = PARAM_PATH + '_final'
            RESULT_DIR = RESULT_DIR + '_final'
    PARAM_PATH = PARAM_PATH + '.pth'
    
    loader = load_test_data(data_path=FLAGS.test_images_path,
        batch_size=1,
        transform=Compose([
            Resize(cons.IMAGE_SIZE,cons.IMAGE_SIZE),
            Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)]))
    model = models.get_model(model_name=FLAGS.model_name,num_classes=cons.NUM_CLASSES)
    if FLAGS.fix_state_dict:
        model.load_state_dict(fix_model_state_dict(torch.load(PARAM_PATH)))
    else:
        model.load_state_dict(torch.load(PARAM_PATH))

    result = inference_loop(model,loader,FLAGS.output_confidence)

    os.makedirs(RESULT_DIR,exist_ok=True)
    result.to_csv(RESULT_DIR+'/inference.csv', index=False)
    if FLAGS.output_confidence:
        result.to_csv(RESULT_DIR+'/inference_with_c.csv', index=False)
        
if __name__ == '__main__':
    flags.mark_flags_as_required(['case','model_name','executed_epoch'])
    app.run(main)