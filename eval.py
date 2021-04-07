import argparse
from io import BytesIO

import mxnet as mx
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.data import FaceImageIter
from infer import FaceAgeGenderModel

parser = argparse.ArgumentParser()
parser.add_argument('--image_size',  default='112,112',         help='models input size.')
parser.add_argument('--model',       default='model/model,200', help='path to load model.')
parser.add_argument('--mtcnn_model', default='mtcnn-model',     help='path to load model.')
parser.add_argument('--gpu',         default=0,   type=int,     help='gpu id')
args = parser.parse_args()


def eval():
    model = FaceAgeGenderModel(args)

    val_dataiter = FaceImageIter(batch_size=1,
                                 data_shape=(3, 112, 112),
                                 path_imgrec="dataset/val.rec",
                                 shuffle=False,
                                 rand_mirror=False,
                                 mean=0)
    gender_correct = 0
    age_correct = 0
    pbar = tqdm(total=len(val_dataiter.seq))
    while True:
        try:
            label, s, _, _ = val_dataiter.next_sample()
            gender = int(label[0])
            age = int(label[1])
            img = val_dataiter.imdecode(s)
            buf = BytesIO()
            img = Image.fromarray(img.asnumpy(), 'RGB')
            img.save(buf, format='JPEG', quality=100)
            buf = buf.getvalue()
            img = Image.open(BytesIO(buf))

            img = np.array(img).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = mx.nd.array(img)
            img = mx.io.DataBatch(data=(img,))

            gender1, age1 = model.get_ga(img)
            if gender == gender1:
                gender_correct += 1
            if age == age1 or age == age1 + 1 or age == age1 + 2 or age == age1 + 3 or \
                    age == age1 - 1 or age == age1 - 2 or age == age1 - 3:
                age_correct += 1
            pbar.update()
        except StopIteration:
            break
    print("性别准确率：%f" % (gender_correct / len(val_dataiter.seq)))
    print("年龄准确率：%f" % (age_correct / len(val_dataiter.seq)))


if __name__ == '__main__':
    eval()
