# CRPN
Siamese Region Proposal Network with Online Updating for Visual Tracking(based on SiamDW(CVPR2019),CRPN(CVPR2019),DROL(AAAI2020))
![image](https://github.com/zhu2014yi/CRPN/blob/master/CRPNv1.png)
## Main Results
| Models  | VOT16 |OTB2015|
| :------ | :------: | :------:|
| SiamDW+depthwise  | 0.387 | -|
| C-RPN     | 0.419 | -|
|C-RPN(online)|0.430|0.683|
### Weights file
[CRPN](https://pan.baidu.com/s/1L22M5bnROjmxWCsOICf4XA )
提取码：vyel
## Anaconda Environment
Please follow the [SiamDW](https://github.com/researchmm/SiamDW)
## Training datasets
Please follow the pysot
### Install pysot toolkit 
cd CRPN;python setup.py build_ext --inplace
## Train&Test
1. conda activate SiamDW(environment)
2. cd CRPN
3. export PYTHONPATH=/home/xxx/CRPN/:PYTHONPATH
4. cd siamese_tracking
5. python onekey.py 
## Eval
python eval.py
## Note
1. When train the SiamDW+depthwise, you can use the default parameters.
2. Eao will lower than origin version(0.393->0.390) due to the number of digits after the decimal point,and the results of OTB2015 may higher than use the original one. 
3. pysot toolkit support VOT2016,2018,2019 and OTB100(2015).
4. pysot toolkit only support python>=3.7.
5. The pretrain_otb.pth is the pretrained model for training the CRPN and test in OTB2015
6. the original pretrain.model is used for training the CRPN and test in VOT2016
7. Tricks: use different seed to train the CRPN and test on the single dataset etc vot2016 to get the highest EAO and then start to tune on the vot2016 may get the better results. the same as OTB2015.
8. The online results is get from the tune.py and different machine may get different results.
## Update
2020/5/9. add the CRPN.yaml config file, you can use the CRPN without change the parameters in SiamRPN.yaml.

2020/6/2. add the pysot toolkit to eval the tracker,usage: 1.cd siamese_tracking;2. python test_siamrpn_main.py.

2020/8/25. change the onekey training way after training finish it will auto execute test from epoch 20-50 and execute eval

2020/8/26. upload weights files

2021/2/17. upload the complete code.
## Acknowledge
pysot SiamDW and DROL
