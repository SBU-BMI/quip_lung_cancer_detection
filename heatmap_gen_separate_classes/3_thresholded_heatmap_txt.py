import os
import sys
import glob
import numpy as np
import multiprocessing as mp

in_fol = '../data/heatmap_txt'
out_fol = '../data/heatmap_txt_6classes_separate_class/heatmap_txt_thresholded'
if not os.path.exists(out_fol):
    os.mkdir(out_fol)

probs = [0.25, 0.1, 0.45, 0.6, 0.75, 0.95]
files = glob.glob(in_fol + '/prediction*')

def process(file):
    print(file)
    slide_id = file.split('/')[-1]
    preds = [f.rstrip().split(' ') for f in open(file, 'r')]
    out = open(os.path.join(out_fol, slide_id), 'w')
    for pred in preds[1:]:
        grades = np.array([float(p) for p in pred[2:]])
        res = probs[np.argmax(grades)] if sum(grades) > 0 else 0
        out.writelines('{} {} {} 0 \n'.format(pred[0], pred[1], res))

    out.close()
    color_fn = 'color-' + slide_id.split('prediction-')[-1]
    os.system('cp {} {}'.format(os.path.join(in_fol, color_fn), os.path.join(out_fol, color_fn)))

print(len(files))
pool = mp.Pool(processes=20)
pool.map(process, files)



