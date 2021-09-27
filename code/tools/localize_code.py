import json

log_path = './test/image/celeba.json'
with open(log_path, 'r') as f:
    log_dic = json.load(f)
addr_dic = log_dic['0.8']
sorted_addr = {k: v for k, v in sorted(addr_dic.items(), key=lambda item: item[1], reverse=True)}

addrs = list(sorted_addr.keys())[:20]
cnts = list(sorted_addr.values())[:20]

addr_cnt = {}

for i in range(len(addrs)):
    k = addrs[i][-3:] + ':'
    if k in addr_cnt.keys():
        addr_cnt[k] += int(cnts[i])
    else:
        addr_cnt[k] = int(cnts[i])

# for i in range(len(addrs)):
#     addrs[i] = addrs[i][-3:] + ':'
# addrs = list(set(addrs))

print(addr_cnt.keys())
dis_file = './image-libturbojpeg.dis'
func_file = './image-libturbojpeg_func'

dis_f = open(dis_file, 'r')
func_f = open(func_file, 'r')

lines_dis = dis_f.readlines()
lines_func = func_f.readlines()

i = 0

result = dict()
result_cnt = dict()

addrs = list(addr_cnt.keys())

for addr in addrs:
    for line_dis in lines_dis:
        chars = line_dis.split('\t')
        if len(chars) > 2 and addr in chars[0] and 'mov' in chars[2]:
            x = int(chars[0][:-1], 16)
            for k in range(len(lines_func) - 1):
                line_func = lines_func[k]
                funcs = line_func.split()
                line_func_1 = lines_func[k + 1]
                funcs_1 = line_func_1.split()
                if x >= int(funcs[0], 16) and x < int(funcs_1[0], 16):
                    if funcs[1] in result.keys():
                        result[funcs[1]].append(chars[0][:-1].strip())
                        result_cnt[funcs[1]] += addr_cnt[addr]
                    else:
                        result[funcs[1]] = [chars[0][:-1].strip()]
                        result_cnt[funcs[1]] = addr_cnt[addr]
                    break

for key in result.keys():
    print(key, result_cnt[key], result[key])