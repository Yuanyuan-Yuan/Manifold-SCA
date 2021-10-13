import urllib
import urllib.request
import urllib.error
import time
import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recons_dir', type=str, default='')
parser.add_argument('--target_dir', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
args = parser.parse_args()


def face_compare(http_url, key, secret, filename1, filename2, max_try=10):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filename1, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file1')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    fr = open(filename2, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file2')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s--\r\n' % boundary)
 
    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')
 
    http_body = b'\r\n'.join(data)
    req = urllib.request.Request(url=http_url, data=http_body)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try_nums = 0
    confidence = -2
    thresholds = None
    while True:
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            qrcont = resp.read()
            mydict = eval(qrcont)
            if len(mydict['faces1']) and len(mydict['faces2']):
                confidence = mydict['confidence']
            break
        except:
            if try_nums > max_try:
                print('Error!')
                confidence = -1
                break
            try_nums += 1
            time.sleep(1)
    return confidence

http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
key = ''
secret = ''

name_list = sorted(os.listdir(args.recons_dir))
total_num = 0
same_num = 0

with open(output_path, 'w') as f:
    for name in tqdm(name_list):
        recons_path = os.path.join(args.recons_dir, name)
        target_path = os.path.join(args.target_dir, name)

        confidence = face_compare(http_url, key, secret, file_path1, file_path2)
        if confidence == -1:
            f.write('%s\t%06d\t%s\n' % (name, i, 'TLE'))
        elif confidence == -2:
            f.write('%s\t%06d\t%s\n' % (name, i, 'non-face'))
        else:
            f.write('%s\t%06d\t%f\n' % (name, i, confidence))

        if confidence > 50:
            correct_num += 1
        total_num += 1
        
        if i % 100 == 0:
            print('Acc: %f' % (correct_num / total_num))
