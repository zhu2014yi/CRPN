# CRPN
Unofficial implementation of CRPN
## Main Results
| Models  | VOT16 |
| :------ | :------: | 
| SiamDW+depthwise  | 0.387 | 
| C-RPN     | 0.395 | 

## Anaconda Environment
Please follow the [SiamDW](https://github.com/researchmm/SiamDW)
## Train&Test
1. conda activate SiamDW(environment)
2. cd CRPN
3. export PYTHONPATH=/home/xxx/CRPN/:PYTHONPATH
4. cd siamese_tracking
5. python onekey.py 
## Note
1. Please set the LR_END in SiamRPN.yaml(experiments/train/SiamRPN.yaml) to 0.000001  and Model:"SiamRPNRes22" to Model:"CascadedSiamRPNRes22" when train the CRPN .
2. When train the SiamDW+depthwise, you can use the default parameters.
## Update
2020/5/9. add the CRPN.yaml config file, you can use the CRPN without change the parameters in SiamRPN.yaml.
