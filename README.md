# 年龄性别识别
年龄性别识别，基于[insightface](https://github.com/deepinsight/insightface)功能模块开发的，支持多张人脸同时检测和识别。

# 环境

 - 安装mxnet，支持1.3~1.6版本，安装命令如下。
```shell script
pip install mxnet-cu101==1.5.1
```

# 数据集

 - 默认支持以下三种数据集，将以下三个数据集下载解压到`dataset`目录下。

1. http://afad-dataset.github.io/
2. http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/
3. https://ibug.doc.ic.ac.uk/resources/agedb/

 - 执行生成数据列表。
```shell script
python create_dataset.py
```

 - 如果想训练自定义数据集，只需生成类似以下的数据列表就可以了。
```shell script

```

如果想看各个年龄的分布，可以执行`show_age_distribution.py`生成年龄分布图。

![年龄分布](https://img-blog.csdnimg.cn/20210407162913663.png)


# 训练

开始训练，具体参数可以查看代码，这里介绍一下`network`参数，这个参数是选择模型的，当指定为`r50`则选择ResNet作为特征提取模型，当`m50`则使用MobileNet作为特征提取模型。
```shell script
python train.py
```

训练输出的结果：
```shell
gpu num: 1
num_layers 50
data_shape [3, 112, 112]
Called with argument: Namespace(batch_size=128, color=0, ctx_num=1, cutoff=0, data_dir='dataset', data_shape='3,112,112', end_epoch=200, gpu_ids='0', image_channel=3, image_h=112, image_w=112, lr=0.1, lr_steps='10,30,80,150,200', network='m50', num_layers=50, prefix='temp/model', pretrained='', rand_mirror=1, rescale_threshold=0, version_input=1, version_output='GAP')
1 GAP 32
INFO:root:loading recordio dataset\train.rec...
INFO:root:dataset\train.rec 数据大小：303018
INFO:root:是否随机翻转图片：1
INFO:root:loading recordio dataset\val.rec...
INFO:root:dataset\val.rec 数据大小：1032
INFO:root:是否随机翻转图片：False
call reset()
开始训练...
INFO:root:Epoch[0] Batch [0-20]	Speed: 520.85 samples/sec	acc=0.572545	MAE=10.734747	CUM_5=0.240699
INFO:root:Epoch[0] Batch [20-40]	Speed: 518.95 samples/sec	acc=0.589844	MAE=9.351172	CUM_5=0.289844
INFO:root:Epoch[0] Batch [40-60]	Speed: 516.86 samples/sec	acc=0.603125	MAE=9.184766	CUM_5=0.303906
INFO:root:Epoch[0] Batch [60-80]	Speed: 508.44 samples/sec	acc=0.609766	MAE=8.759375	CUM_5=0.336719
INFO:root:Epoch[0] Batch [80-100]	Speed: 461.26 samples/sec	acc=0.656250	MAE=8.224609	CUM_5=0.361328
INFO:root:Epoch[0] Batch [100-120]	Speed: 518.43 samples/sec	acc=0.696875	MAE=7.611328	CUM_5=0.400391
INFO:root:Epoch[0] Batch [120-140]	Speed: 514.88 samples/sec	acc=0.715234	MAE=7.224609	CUM_5=0.426172
INFO:root:Epoch[0] Batch [140-160]	Speed: 517.80 samples/sec	acc=0.722266	MAE=6.976172	CUM_5=0.437500
```

# 评估

训练结束之后，执行下面的命令评估模型的识别准确率。
```shell script
python eval.py
```

输出结果，从结果来看，准确率还是可以的。
```shell
100%|██████████| 1032/1032 [00:06<00:00, 153.75it/s]
性别准确率：0.972868
年龄准确率：0.761628
```

# 预测

使用训练好的模型或者笔者提供的模型执行年龄性别识别，通过指定图像文件路径完成识别。
```shell script
python infer.py --image=test.jpg
```

识别输出结果：
```shell
第1张人脸，位置(160, 32, 204, 84), 性别：男, 年龄：30
第2张人脸，位置(545, 162, 579, 206), 性别：女, 年龄：31
第3张人脸，位置(632, 118, 666, 158), 性别：男, 年龄：28
第4张人脸，位置(91, 159, 151, 237), 性别：男, 年龄：38
第5张人脸，位置(723, 123, 760, 169), 性别：男, 年龄：26
第6张人脸，位置(263, 120, 317, 191), 性别：男, 年龄：27
第7张人脸，位置(438, 134, 481, 190), 性别：男, 年龄：46
第8张人脸，位置(908, 160, 963, 224), 性别：男, 年龄：35
第9张人脸，位置(39, 51, 81, 102), 性别：女, 年龄：31
第10张人脸，位置(807, 148, 847, 196), 性别：女, 年龄：26
第11张人脸，位置(449, 40, 485, 84), 性别：男, 年龄：29
第12张人脸，位置(378, 46, 412, 86), 性别：女, 年龄：33
第13张人脸，位置(534, 46, 567, 83), 性别：男, 年龄：30
第14张人脸，位置(272, 20, 311, 67), 性别：男, 年龄：28
第15张人脸，位置(358, 216, 375, 237), 性别：男, 年龄：27
```

效果图：
![识别结果](https://img-blog.csdnimg.cn/20210407165918268.jpg)