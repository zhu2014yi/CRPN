# CRPN
Unofficial implementation of CRPN
## Main Results
| Models  | VOT16 |
| :------ | :------: | 
| SiamDW+depthwise  | 0.387 | 
| C-RPN     | 0.395 | 
### Weights file
[CRPN](https://pan.baidu.com/s/1L22M5bnROjmxWCsOICf4XA )
提取码：vyel
## Anaconda Environment
Please follow the [SiamDW](https://github.com/researchmm/SiamDW)

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
## Update
2020/5/9. add the CRPN.yaml config file, you can use the CRPN without change the parameters in SiamRPN.yaml.

2020/6/2. add the pysot toolkit to eval the tracker,usage: 1.cd siamese_tracking;2. python test_siamrpn_main.py.

2020/8/25. change the onekey training way after training finish it will auto execute test from epoch 20-50 and execute eval

2020/8/26. upload weights files
## Acknowledge
pysot and SiamDW
