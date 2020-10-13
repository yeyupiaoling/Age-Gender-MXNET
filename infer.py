import argparse
import cv2
import mxnet as mx
import numpy as np
import face_preprocess
from data import FaceImageIter
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from mtcnn_detector import MtcnnDetector

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='temp/8DC8CBE1.png', help='')
parser.add_argument('--model', default='model/model,0', help='path to load model.')
parser.add_argument('--mtcnn-model', default='mtcnn-model', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()


class FaceAgeGenderModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu()
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model) > 0:
            self.model = self.get_model(ctx, image_size, args.model, 'fc1')

        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_size = image_size
        if args.det == 0:
            detector = MtcnnDetector(model_folder=args.mtcnn_model, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=args.mtcnn_model, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        print("加载模型：%s" % args.mtcnn_model)
        self.detector = detector

    def get_model(self, ctx, image_size, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model

    def get_face(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        return db

    def get_ga(self, data):
        self.model.forward(data, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age


def main():
    model = FaceAgeGenderModel(args)
    img = cv2.imread(args.image)
    img = model.get_face(img)
    print(img)
    if img is not None:
        gender, age = model.get_ga(img)
        print('gender is', gender)
        print('age is', age)


def eval_val():
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
            if age == age1 or age == age1 + 1 or age == age1 + 2 or age == age1 + 3 or\
                    age == age1 - 1 or age == age1 - 2 or age == age1 - 3:
                age_correct += 1
            pbar.update()
        except StopIteration:
            break
    print("性别准确率：%f" % (gender_correct / len(val_dataiter.seq)))
    print("年龄准确率：%f" % (age_correct / len(val_dataiter.seq)))


if __name__ == '__main__':
    # main()
    eval_val()
