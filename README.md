# Spatial Pooling for Torch7

This repositery proposed the implementation of several spatial poolings used for weakly supervised learning of deep ConvNets.


## Installation

```
$ git clone https://github.com/durandtibo/spatial-pooling.torch.git
$ cd spatial-pooling.torch
$ luarocks make rocks/spatial-pooling-scm-1.rockspec
```
To test the installation, you can run
```
$ th test/test.lua
```

## Modules

* [GlobalMaxPooling](#nn.GlobalMaxPooling)
* [GlobalkMaxPooling](#nn.GlobalkMaxPooling)
* [GlobalAveragePooling](#nn.GlobalMaxPooling)
* [LogSumExpPooling](#nn.LogSumExpPooling)
* [WeldonPooling](#nn.WeldonPooling)

<a name="nn.GlobalMaxPooling"></a>
### GlobalMaxPooling (GMP) ###

Global Max Pooling is a spatial pooling strategy used in ["Is object localization for free? – Weakly Supervised Object Recognition with Convolutional Neural Networks "](http://www.di.ens.fr/willow/research/weakcnn/).

```lua
module = nn.GlobalMaxPooling()
```

Applies 2D max-pooling operation on the whole image. The number of output features is equal to the number of input planes.

If the input image is a 4D tensor `nBatchImage x nInputPlane x w x h`, the output image size will be `nBatchImage x nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

If the input image is a 3D tensor `nInputPlane x w x h`, the output image size will be `nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.



##### Equation
We note `z^c` the c-th map of input, and `s^c` the c-th map of the output.
```
s^c = max_{i,j} z^c_{i,j}
```

##### References

```
@inproceedings{Oquab_DeepMIL_CVPR15,
author = "Oquab, M. and Bottou, L. and Laptev, I. and Sivic, J.",
title = "Is object localization for free? – Weakly-supervised learning with convolutional neural networks",
booktitle =  "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition",
year = "2015"
}
```

<a name="nn.GlobalkMaxPooling"></a>
### GlobalkMaxPooling ###

GlobalkMaxPooling is a generalization of GlobalMaxPooling to multiple maximums. This spatial pooling strategy is inspired from Top Instances model ["Multiple Instance Learning for Soft Bags via Top Instances"](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Multiple_Instance_Learning_2015_CVPR_paper.pdf).

```lua
module = nn.GlobalkMaxPooling(kMax)
```
Applies 2D k-max-pooling operation on the whole image. The number of output features is equal to the number of input planes.


The parameter is the following:
  * `kMax`: The number of top instances. `kMax` can defined the number of selected regions (`kMax >= 1`) or the proportion of selected regions (`0 < kMax < 1`). If `kMax <= 0`, all the regions are selected.
  Default is `kMax = 1`.

If the input image is a 4D tensor `nBatchImage x nInputPlane x w x h`, the output image size will be `nBatchImage x nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

If the input image is a 3D tensor `nInputPlane x w x h`, the output image size will be `nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.


##### Equation
We note `z^c` the c-th map of input, and `s^c` the c-th map of the output.
```
s^c = max_{h in H_kMax} 1 / kMax sum_{i,j} h_{i,j} z^c_{i,j}
```
where `H_k` is such that `h in H_k` satisfies `h_{i,j} in {0, 1}` and `sum_{i,j} h_{i,j} = k`

##### Special cases
* GlobalMaxPooling: if `kMax = 1`
* GlobalAveragePooling: if `kMax = h x w`


<a name="nn.GlobalAveragePooling"></a>
### GlobalAveragePooling (GAP) ###

Global Max Pooling is a spatial pooling strategy used in ["Learning Deep Features for Discriminative Localization "](http://cnnlocalization.csail.mit.edu/).

```lua
module = nn.GlobalAveragePooling()
```

Applies 2D average-pooling operation on the whole image. The number of output features is equal to the number of input planes.

If the input image is a 4D tensor `nBatchImage x nInputPlane x w x h`, the output image size will be `nBatchImage x nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

If the input image is a 3D tensor `nInputPlane x w x h`, the output image size will be `nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

##### Equation
We note `z^c` the c-th map of input, and `s^c` the c-th map of the output.
```
s^c = 1 / (h * w) sum_{i,j} z^c_{i,j}
```

#### References
```
@inproceedings{Zhou_2016_CVPR,
author = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
title = {{Learning Deep Features for Discriminative Localization}},
booktitle = {CVPR},
year = {2016}
}
```


<a name="nn.LogSumExpPooling"></a>
### LogSumExpPooling ###

LogSumExpPooling is a spatial pooling strategy used in ["From Image-level to Pixel-level Labeling with Convolutional Networks "](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Pinheiro_From_Image-Level_to_2015_CVPR_paper.pdf).

```lua
module = nn.LogSumExpPooling(beta)
```

Applies 2D LogSumExp-pooling operation on the whole image. The number of output features is equal to the number of input planes.

The parameter is the following:
  * `beta`: The anti-temperature parameter. Default is `beta=1`.

If the input image is a 4D tensor `nBatchImage x nInputPlane x w x h`, the output image size will be `nBatchImage x nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

If the input image is a 3D tensor `nInputPlane x w x h`, the output image size will be `nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

##### Equation
We note `z^c` the c-th map of input, and `s^c` the c-th map of the output.
```
s^c = log 1 / (h * w) sum_{i,j} exp(beta * z^c_{i,j})
```

##### Special cases
* GlobalMaxPooling: if `beta = +inf`
* GlobalAveragePooling: if `beta = 0`


#### References
```
@inproceedings{pinheiro_weak_seg_cvpr15,
Author = {Pedro O. Pinheiro and Ronan Collobert},
Title = {{From Image-level to Pixel-level Labeling with Convolutional Networks}},
booktitle = {CVPR},
Year = {2015}
}
```


<a name="nn.WeldonPooling"></a>
### WeldonPooling ###

WeldonPooling is a spatial pooling module used in ["WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks"](http://webia.lip6.fr/~durandt/pdfs/2016_CVPR/Durand_WELDON_CVPR_2016.pdf).

```lua
module = nn.WeldonPooling(kMax, kMin)
```

Applies 2D WELDON-pooling operation on the whole image. The number of output features is equal to the number of input planes.

The parameters are the following:
  * `kMax`: The number of top instances. It is possible to define the number of selected regions (`kMax >= 1`) or the proportion of selected regions (`0 <= kMax < 1`). If `kMax < 0`, `kMax` is set to `0`. Default is `kMax = 1`.
  * `kMin`: The number of low instances. It is possible to define the number of selected regions (`kMin >= 1`) or the proportion of selected regions (`0 <= kMin < 1`). If `kMin < 0`, `kMin` is set to `0`. Default is `kMin = 1`.

If the input image is a 4D tensor `nBatchImage x nInputPlane x w x h`, the output image size will be `nBatchImage x nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

If the input image is a 3D tensor `nInputPlane x w x h`, the output image size will be `nInputPlane x 1 x 1` where `w` and `h` are spatial image dimensions.

##### Equation
We note `z^c` the c-th map of input, and `s^c` the c-th map of the output.
```
s^c = max_{h in H_kMax} 1 / kMax sum_{i,j} h_{i,j} z^c_{i,j} + min_{h in H_kMin} 1 / kMin sum_{i,j} h_{i,j} z^c_{i,j}
```
where `H_k` is such that `h in H_k` satisfies `h_{i,j} in {0, 1}` and `sum_{i,j} h_{i,j} = k`


##### Special cases
* GlobalMaxPooling: if `kMax = 1` and `kMin = 0`
* GlobalAveragePooling: if `kMax = h x w` and `kMin = 0`
* MantraPooling: if `kMax = 1` and `kMin = 1`

##### References

```
@inproceedings{Durand_WELDON_CVPR_2016,
author = {Durand, Thibaut and Thome, Nicolas and Cord, Matthieu},
title = {{WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2016}
}
```


## Licence

MIT License
