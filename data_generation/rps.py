import os
import requests
import zipfile
import errno


def download_rps(localfolder='/zdata/Github/pytorchsbs'):
    filenames = ['rps.zip', 'rps-test-set.zip']
    for filename in filenames:
        try:
            if not os.path.exists(localfolder):
                os.mkdir(localfolder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(f'{localfolder} already exists!')
            else:
                raise

        localfile = f'{localfolder}/{filename}'
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/{}'
        req = requests.get(url.format(filename), allow_redirects=True)
        with open(localfile, 'wb') as f:
            f.write(req.content)

        with zipfile.ZipFile(localfile, 'r') as zip_ref:
            zip_ref.extractall(localfolder)
