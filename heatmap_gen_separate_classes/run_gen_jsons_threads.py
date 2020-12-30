import os
import sys

def create_new_fol(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)

out_dir = os.environ.get('OUT_DIR')
if out_dir is None:
   out_dir = "/root/quip_lung_cancer_detection/data" 

parent_in = str(out_dir)+'/heatmap_txt_6classes_separate_class'
#seperate_in = ['heatmap_txt_lepidic', 'heatmap_txt_benign', 'heatmap_txt_acinar','heatmap_txt_micropap', 'heatmap_txt_mucinous', 'heatmap_txt_solid', 'heatmap_txt_thresholded']
seperate_in = [fol for fol in os.listdir(parent_in) if len(fol) > 5]

log_fol = str(out_dir)+'/log/'
heatmap_versions = ['lung-john-6c_' + f.split('_')[-1] + '_20200201_hanle' for f in seperate_in]
heatmap_txt_in = [os.path.join(parent_in, fol) for fol in seperate_in]

for version, txt in zip(heatmap_versions, heatmap_txt_in):
    log_file = log_fol + 'log.heatmap_jsons_' + txt.split('/')[-1]
    cmd = 'nohup bash gen_all_json.sh {} {} &> {} &'.format(version, txt, log_file)
    print(cmd)
    os.system(cmd)

