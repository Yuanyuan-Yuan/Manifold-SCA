import os
import numpy as np
import json
import progressbar

dataset = 'Sub-URMP'
sub = 'test'
grad_dir = ('/root/output/gradient_%s/%s_cacheline/' % (sub, dataset))
inst_dir = ('/root/data/localize/pin_raw/%s/%s/' % (dataset, sub))
result_path = ('/root/output/pinpoint/%s/' % (sub))

grad_list = sorted(os.listdir(grad_dir))[:500]
inst_list = sorted(os.listdir(inst_dir))[:500]


print('Dataset: %s' % dataset)

result_dic = {}

threshold_list = [0.6, 0.8, 0.9]
for threshold in threshold_list:
    result_dic[threshold] = {}

progress = progressbar.ProgressBar(maxval=len(inst_list)).start()
for i, inst_name in enumerate(inst_list):
    progress.update(i + 1)
    grad_name = inst_name.split('.')[0] + '-grad.npz'
    #assert inst_name.split('.') == grad_name.split('-')
    grad = np.load(grad_dir + grad_name)['arr_0']
    with open(inst_dir + inst_name, 'r') as f:
        inst = f.readlines()
        inst = inst[:-1]
    for threshold in threshold_list:
        index = np.where((grad > threshold) == 1)[0]
        selected_addr = []
        for idx in index:
            if idx >= len(inst):
                continue
            selected_inst = inst[idx]
            op_addr = selected_inst.split(' ')[0]
            op_addr = op_addr[:-1]
            selected_addr.append(op_addr)
        for addr in selected_addr:
            if addr in result_dic[threshold].keys():
                result_dic[threshold][addr] += 1
            else:
                result_dic[threshold][addr] = 1
progress.finish()

with open(('%s%s.json' % (result_path, dataset)), 'w') as f:
    json.dump(result_dic, f)