# Converting python based VAE code to C++

This is an attempt to converting pytorch based code to libtorch based code.

Used [Basic VAE Example from pytorch/examples](https://github.com/pytorch/examples/tree/master/vae) as a base code.

**The functionality of this code is not yet verified. DO NOT USE IT AS REFERENCE.**

## Environment

- CentOS 7.7.1908
- Developer Toolset 7
- Python-3.7.7
- cuda-10.2.89_440.33.01
- cudnn7-7.6.5.33-1.cuda10.2.x86_64

## Tips

- libtorch from [official pytorch downloads](https://pytorch.org/get-started/locally/) didn't work for me.
- clone source from [pytorch repository](https://github.com/pytorch/pytorch) and run ```setup.py``` with ```build``` option.
- use ```include``` and ```lib``` from the build as your libtorch. 

## Reference

- [libtorch example from [TonyzBi/libtorch-mnist]](https://github.com/TonyzBi/libtorch-mnist)
