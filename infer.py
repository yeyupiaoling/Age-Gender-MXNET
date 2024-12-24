import argparse

import cv2
import mxnet as mx
import numpy as np

from utils import face_preprocess
from utils.mtcnn_detector import MtcnnDetector

parser = argparse.ArgumentParser()
parser.add_argument('--image_size',   default='112,112',           help='models input size.')
parser.add_argument('--image',        default='test.jpg',          help='infer image path.')
parser.add_argument('--model',        default='model/model,200',   help='path to load model.')
parser.add_argument('--mtcnn_model',  default='mtcnn-model',       help='path to load model.')
parser.add_argument('--gpu',          default=0, type=int,         help='gpu id')
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
        detector = MtcnnDetector(model_folder=args.mtcnn_model, ctx=ctx, num_worker=1, accurate_landmark=True,
                                 threshold=self.det_threshold)
        print("加载模型：%s" % args.mtcnn_model)
        self.detector = detector

    # 加载模型
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

    # 识别人脸
    def get_faces(self, face_img):
        ret = self.detector.detect_face(face_img)
        if ret is None:
            return [], [], []
        bbox, points = ret
        if bbox.shape[0] == 0:
            return [], [], []
        bboxes = []
        pointses = []
        faces = []
        for i in range(len(bbox)):
            b = bbox[i, 0:4]
            bboxes.append(b)
            p = points[i, :].reshape((2, 5)).T
            pointses.append(p)
            nimg = face_preprocess.preprocess(face_img, b, p, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            faces.append(db)
        return faces, bboxes, pointses

    # 性别年龄识别
    def get_ga(self, data):
        self.model.forward(data, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age


# 画出人脸框和关键点
def draw_face(image_path, bboxes, pointses, genderes, ages):
    img = cv2.imread(image_path)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        # 判别为人脸的置信度
        cv2.putText(img, '{},{}'.format('m' if genderes[i] == 0 else 'f', ages[i]),
                    (corpbbox[0], corpbbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 画关键点
    for i in range(len(pointses)):
        points = pointses[i]
        for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255))
    cv2.imwrite('result.jpg', img)
    cv2.imshow('result', img)
    cv2.waitKey(0)


def main():
    # 加载模型
    model = FaceAgeGenderModel(args)
    # 读取图片
    img = cv2.imread(args.image)
    # 检测人脸
    faces, bboxes, pointses = model.get_faces(img)
    genderes = []
    ages = []
    for i in range(len(faces)):
        # 识别性别年龄
        gender, age = model.get_ga(faces[i])
        print('第%d张人脸，位置(%d, %d, %d, %d), 性别：%s, 年龄：%d' % (i + 1, bboxes[i][0], bboxes[i][1], bboxes[i][2],
                                                           bboxes[i][3], '女' if gender == 0 else '男', age))
        genderes.append(gender)
        ages.append(age)
    draw_face(args.image, bboxes, pointses, genderes, ages)


if __name__ == '__main__':
    main()
