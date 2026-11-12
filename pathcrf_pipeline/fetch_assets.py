"""Restore pinned upstream source and weights without accessing match data."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request

from . import ASSETS

COMMIT='1b7c6f85a6b78c92e4543fde72ca2b86a933f997'
YOLO_COMMIT='5e83fafa8d564243001ce8e063612a618a138fbe'
YOLO_SHA256='69c652bfa9814ef882c439617f04b8fd5749b6b8455aaa3c36110bc2e802aadd'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'PathCRF-hackathon-integration'})
    with urllib.request.urlopen(request,timeout=120) as response:
        return response.read()


def write_verified(path,data,sha,git_blob=True):
    actual=(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_blob else hashlib.sha256(data).hexdigest())
    if actual!=sha:
        raise ValueError('Asset integrity failure: '+str(path))
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('Refusing to overwrite changed file: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-detector',action='store_true')
    args=parser.parse_args()
    tree=json.loads(fetch(f'https://api.github.com/repos/hyunsungkim-ds/pathcrf/git/trees/{COMMIT}?recursive=1'))
    source=[e for e in tree['tree'] if e['type']=='blob' and
            ((e['path'].startswith(('models/','datatools/','set_transformer/')) and e['path'].endswith('.py'))
             or e['path'] in ['inference.py','requirements.txt','LICENSE'])]
    weights=[e for e in tree['tree'] if e['type']=='blob' and e['path'].startswith('saved/140/')]
    def download(entry):
        destination=(ASSETS/entry['path'] if entry in weights else ASSETS/'upstream'/entry['path'])
        data=fetch(f'https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/{COMMIT}/{entry["path"]}')
        write_verified(destination,data,entry['sha'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download,source+weights))
    (ASSETS/'upstream'/'SOURCE.json').write_text(json.dumps({'repository':'https://github.com/hyunsungkim-ds/pathcrf','commit':COMMIT,'files':source},indent=2))
    print(f'Verified {len(source)} source files and {len(weights)} checkpoint files')
    if args.with_detector:
        folder=ASSETS/'detector'
        name='yolo-football-player-detection.pt'
        url=f'https://huggingface.co/martinjolif/yolo-football-player-detection/resolve/{YOLO_COMMIT}'
        data=fetch(f'{url}/{name}')
        write_verified(folder/name,data,YOLO_SHA256,git_blob=False)
        (folder/'MODEL_CARD.md').write_bytes(fetch(f'{url}/README.md'))
        (folder/'SOURCE.json').write_text(json.dumps({'repository':'martinjolif/yolo-football-player-detection','commit':YOLO_COMMIT,'filename':name,'size_bytes':len(data),'sha256':YOLO_SHA256},indent=2))
        print('Verified goalkeeper detector checkpoint')


if __name__=='__main__':
    main()

