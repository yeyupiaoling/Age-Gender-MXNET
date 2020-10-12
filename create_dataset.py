from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import mxnet as mx
import numpy as np
from tqdm import tqdm
from mtcnn_detector import MtcnnDetector
import random
import face_preprocess


class FaceModel:
    def __init__(self, det):
        self.det = det
        ctx = mx.gpu(0)
        self.det_threshold = [0.6, 0.7, 0.8]
        mtcnn_path = 'mtcnn-model'
        if det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        print("加载模型：%s" % mtcnn_path)
        self.detector = detector

    def get_face(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        return nimg


class AgeGenderModel:
    def __init__(self, image_size):
        ctx = mx.gpu(0)
        model_path = 'model/model,0'
        self.model = self.get_model(ctx, image_size, model_path, 'fc1')
        print("加载模型：%s" % model_path)

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

    def get_ga(self, nimg):
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))

        self.model.forward(db, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age


def create_face(image_dir):
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    faceModel = FaceModel(0)
    images = []
    for root, dirs, files in os.walk(image_dir):
        for image in files:
            if image[-3:] != 'jpg':
                continue
            image = os.path.join(root, image)
            images.append(image)
    for image in tqdm(images):
        save_path = os.path.join("dataset", image)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        img = cv2.imread(image)
        if img is not None:
            img1 = faceModel.get_face(img)
            if img1 is not None:
                cv2.imwrite(save_path, img1)


def create_record(output_path, list_paths):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_writer = mx.recordio.MXIndexedRecordIO(os.path.join(output_path, 'train.idx'),
                                                 os.path.join(output_path, 'train.rec'), 'w')
    val_writer = mx.recordio.MXIndexedRecordIO(os.path.join(output_path, 'val.idx'),
                                               os.path.join(output_path, 'val.rec'), 'w')
    train_widx = 0
    val_widx = 0
    for list_path in list_paths:
        with open(list_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
        pbar = tqdm(total=len(lines))
        for i, line in enumerate(lines):
            img_path, gender, age = line.split(',')
            gender = int(gender)
            age = int(age)
            nlabel = [gender, age]
            try:
                if not os.path.exists(img_path):
                    continue
                img = mx.image.imread(img_path).asnumpy()
                if i % 20 == 0:
                    nheader = mx.recordio.IRHeader(0, nlabel, val_widx, 0)
                    s = mx.recordio.pack_img(nheader, img)
                    val_writer.write_idx(val_widx, s)
                    val_widx += 1
                else:
                    if 60 >= age >= 40:
                        repetition = 3
                    elif 40 > age >= 28:
                        repetition = 2
                    else:
                        repetition = 1
                    for _ in range(repetition):
                        nheader = mx.recordio.IRHeader(0, nlabel, train_widx, 0)
                        s = mx.recordio.pack_img(nheader, img)
                        train_writer.write_idx(train_widx, s)
                        train_widx += 1
            except:
                print("图片 %s 发生错误！" % img_path)
            pbar.update()


def create_agedb_list(images_dir, list_path):
    f = open(list_path, 'w', encoding="utf-8")
    images = os.listdir(images_dir)
    for image in tqdm(images):
        img_path = os.path.join(images_dir, image).replace('\\', '/')
        name = image[:-4]
        _, _, age, gender = name.split('_')
        age = int(age)
        if gender == 'f':
            gender = 0
        else:
            gender = 1
        f.write("%s,%d,%d\n" % (img_path, gender, age))
    f.close()


def create_megaage_asian_list(path, list_path):
    ageGenderModel = AgeGenderModel((112, 112))
    f = open(list_path, 'w', encoding="utf-8")
    with open(os.path.join(path, "list", "test_age.txt"), 'r', encoding='utf-8') as f1:
        test_age = f1.readlines()
    with open(os.path.join(path, "list", "test_name.txt"), 'r', encoding='utf-8') as f2:
        test_name = f2.readlines()
    with open(os.path.join(path, "list", "train_age.txt"), 'r', encoding='utf-8') as f3:
        train_age = f3.readlines()
    with open(os.path.join(path, "list", "train_name.txt"), 'r', encoding='utf-8') as f4:
        train_name = f4.readlines()
    for i, name in tqdm(enumerate(test_name)):
        img_path = os.path.join(path, "test", name).replace('\n', '').replace('\\', '/')
        if not os.path.exists(img_path):
            continue
        age = int(test_age[i].replace('\n', ''))
        img = cv2.imread(img_path)
        gender, _ = ageGenderModel.get_ga(img)
        f.write("%s,%d,%d\n" % (img_path, int(gender), age))
    for i, name in tqdm(enumerate(train_name)):
        img_path = os.path.join(path, "train", name).replace('\n', '').replace('\\', '/')
        if not os.path.exists(img_path):
            continue
        age = int(train_age[i].replace('\n', ''))
        img = cv2.imread(img_path)
        gender, _ = ageGenderModel.get_ga(img)
        f.write("%s,%d,%d\n" % (img_path, int(gender), age))
    f.close()


def create_afad_list(images_dir, list_path):
    f = open(list_path, 'w', encoding="utf-8")
    ages_path = os.listdir(images_dir)
    for age in tqdm(ages_path):
        age_path = os.path.join(images_dir, age)
        images_path = os.path.join(age_path, "111")
        images = os.listdir(images_path)
        gender = 1
        for image in tqdm(images):
            img_path = os.path.join(images_path, image).replace('\\', '/')
            f.write("%s,%d,%d\n" % (img_path, gender, int(age)))

        images_path = os.path.join(age_path, "112")
        images = os.listdir(images_path)
        gender = 0
        for image in tqdm(images):
            img_path = os.path.join(images_path, image).replace('\\', '/')
            f.write("%s,%d,%d\n" % (img_path, gender, int(age)))
    f.close()


if __name__ == '__main__':
    create_face("AgeDB")
    create_face("megaage_asian")
    create_face("AFAD")
    create_agedb_list("dataset/AgeDB", "dataset/agedb_list.txt")
    create_megaage_asian_list("dataset/megaage_asian", "dataset/megaage_asian_list.txt")
    create_afad_list("dataset/AFAD", "dataset/afad_list.txt")
    create_record("dataset", ["dataset/agedb_list.txt",
                              "dataset/megaage_asian_list.txt",
                              "dataset/afad_list.txt"])
