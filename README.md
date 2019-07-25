# chinese_ocr
yolo3 + densenet ocr

# setup 
see setup

# dowon model 
url：https://pan.baidu.com/s/1gm0Uq_sLe00En-IbUPiQUg 
password ：qcco 

put the model file in project_root/chinese_ocr/models/densenet_base_model/1

# test
`python demo.py`

you can also see [understand_detect](https://github.com/bing1zhi2/chinese_ocr/blob/master/chinese_ocr/understand_detect.ipynb)

![result](https://github.com/bing1zhi2/chinese_ocr/blob/master/chinese_ocr/test_result/result.png "result")
# train
`cd train`

`python train.py`

## train on your own dataset
put your dataset into train/images and change the label file data_test.txt data_train.txt

# things to do
1. use pretrain model to detect word
   * add demo   &radic;
   * add densenet training code &radic;
 
2. add framework to easy train on your own dataset
   * add yolo3 train code
   * make the code can be  easy use on other dataset
 
  
   
# Reference
https://github.com/chineseocr/chineseocr
https://github.com/YCG09/chinese_ocr
