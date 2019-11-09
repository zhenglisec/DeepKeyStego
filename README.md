# DeepKeyStego
The pytorch implement of paper "DeepKeyStego: Protecting Communication by Key-dependent Steganography with Deep Networks"

# Quickstart
* Configure the environment
```
Python=3.7
Pytorch=1.2.0
Others are the latest version
```
* Change `args.type='symmeric or asymmeric'` to define the implement mode and `args.secret_len=1024 or 8192` to define the secret message length.

# Citation
If you find DeepKeyStego useful in your research, please consider to cite the papers:
```
@inproceedings{li2019deepkeystego,
  title={DeepKeyStego: Protecting Communication by Key-Dependent Steganography with Deep Networks},
  author={Li, Zheng and Han, Ge and Guo, Shanqing and Hu, Chengyu},
  booktitle={2019 IEEE 21st International Conference on High Performance Computing and Communications; IEEE 17th International  Conference on Smart City; IEEE 5th International Conference on Data Science and Systems (HPCC/SmartCity/DSS)},
  pages={1937--1944},
  year={2019},
  organization={IEEE}
}
```
