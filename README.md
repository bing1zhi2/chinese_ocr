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
or you can use [train_with_param](https://github.com/bing1zhi2/chinese_ocr/blob/master/chinese_ocr/train_use_new_dataset.py) to deal with different dataset

## dataset format


&ensp;---dataset    
&ensp;&ensp;&ensp;&ensp;--images    
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;--xxx.jpg   
&ensp;&ensp;&ensp;&ensp;--data_train.txt   
&ensp;&ensp;&ensp;&ensp;--data_test.txt   



## train on dataset
the dataset is uploading ...    
the dataset contains 800,000 pictures 
300,000 from chinese novel  
100,000 from random number 0-9    
100,000 from random code    
300,000 random selected by it's frequency    

* Random char space
* Random font size 
* 10 different fonts
* Blur
* noise(gauss,uniform,salt_pepper,poisson)
* ...


Uploading takes times....

Or you can use YCG09's dataset to train,url:

url：https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw (passwd：lu7m)


put your dataset into train/images and change the label file data_test.txt data_train.txt


## generate you own dataset
or you can generate your own dataset:
### text location:
[SynthText](https://github.com/JarveeLee/SynthText_Chinese_version) 
### text recognition
[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)   
[text_renderer](https://github.com/Sanster/text_renderer)  (which one I used )
you can use tools/tmp_label_to_id_label.py to change label file format to what we need here


#  update
1. use pretrain model to detect word
   * add demo   &radic;
   * add densenet training code &radic;
   * test gpu nms &radic;
   * generate my own dataset &radic;
 
2. add framework to easy train on your own dataset
   * add yolo3 train code
   * make the code can be  easy use on other dataset
 
  
   
# Reference
https://github.com/chineseocr/chineseocr
https://github.com/YCG09/chinese_ocr
