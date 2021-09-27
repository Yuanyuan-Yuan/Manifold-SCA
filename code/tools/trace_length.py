import os
import progressbar

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

dataset_list = ['celeba', 'chestX-ray8', 'COCO_caption', 'DailyDialog', 'SC09', 'Sub-URMP']

for dataset in dataset_list:
    raw_dir = ('/root/dataset/pin_raw/%s/test/' % dataset)
    name_list = sorted(os.listdir(raw_dir))
    length_list = []
    progress = progressbar.ProgressBar(maxval=len(name_list), widgets=widgets).start()
    for i, name in enumerate(name_list):
        progress.update(i + 1)
        with open(raw_dir + name, 'r') as f:
            length_list.append(len(f.readlines()) - 1)
    progress.finish()
    print('%s mean: %f' % (dataset, sum(length_list) / len(length_list)))
    print('%s max: %d' % (dataset, max(length_list)))
    print('%s min: %d' % (dataset, min(length_list)))