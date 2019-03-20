# Face parsing

### **数据集**

采用公开数据集 Helen

包含 11类标签： hair,eyebrows, eyes, nose, lips, in mouth, skin and background.

包含 194个人脸关键点，training: 2000   val: 330

将训练集做随机反转、旋转、剪切进行数据扩充，training: 10146  val: 330

### **网络搭建**

以UNet为基础网络框架，将编码器部分改为resnet18和mobilenet ，使用在imagenet上的预训练模型，在解码器部分添加attention gate，训练准确率有提升。

overall acc表示11个类别的平均准确率

| Model          | Overall acc(Tr) | Overall acc(val) | mIOU (tr)  | mIOU (val) | Params (M) |
| -------------- | --------------- | ---------------- | ---------- | ---------- | ---------- |
| UNet           | 0.9292          | 0.9033           | 0.6142     | 0.5315     | 118.49     |
| Resnet18       | 0.9679          | 0.9274           | 0.7885     | 0.6123     | 69.94      |
| Resnet18+ImN   | 0.9697          | 0.9296           | 0.8016     | 0.6402     | 69.94      |
| mobile         | 0.9466          | 0.9248           | 0.6787     | 0.6177     | 17.17      |
| **mobile+ImN** | **0.9689**      | **0.9334**       | **0.7976** | **0.6969** | **17.17**  |

ImN: initializing the network from the weights pretrained on ImageNet dataset.



UNet (mobilenet encoder + ImN) per class acc

| classname | background | face  | left eyebrow | right eyebrow | left eye | right eye | nose  | upper lip | inner mouth | lower lip | hair  |
| --------- | ---------- | ----- | ------------ | ------------- | -------- | --------- | ----- | --------- | ----------- | --------- | ----- |
| acc       | 0.962      | 0.932 | 0.727        | 0.737         | 0.800    | 0.783     | 0.935 | 0.751     | 0.803       | 0.812     | 0.790 |
| miou      | 0.930      | 0.875 | 0.570        | 0.559         | 0.589    | 0.615     | 0.849 | 0.596     | 0.598       | 0.675     | 0.663 |

### 训练结果

![](./img/faceseg.gif)

