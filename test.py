import argparse
import datetime

import cv2

import face_model

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='temp/8DC8CBE1.png', help='')
parser.add_argument('--model', default='model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread(args.image)
img = model.get_input(img)
if img is not None:
    gender, age = model.get_ga(img)
    time_now2 = datetime.datetime.now()
    print('gender is', gender)
    print('age is', age)
