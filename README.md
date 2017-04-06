# Mask RCNN in PyTorch
1. This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of [Mask RCNN](https://arxiv.org/abs/1703.06870) which attempts to reproduce the results in [Mask RCNN](https://arxiv.org/abs/1703.06870).
2. This project provides an implementation of ROI Align in CUDA C as well as a PyTorch nn.Module for it.
3. The model architecture is based on the awesome [Faster RCNN with PyTorch](https://github.com/longcw/faster_rcnn_pytorch) repo.
4. The experiment setup is based on [Image Classification Project Killer in Pytorch](https://github.com/felixgwu/img_classification_pk_pytorch) to minimize the effort of doing experiments and developing new models.

Spectial thanks to [Fast Mask RCNN](https://github.com/CharlesShang/FastMaskRCNN) for being the catalyst of this project.

## Progress (with expected time)
- [x] Implement ROIAlign layer
- [x] COCO dataloader with mask
- [ ] FastRCNN with ROIAlign using ResNet-50-C4 (4/6)
- [ ] Training code (4/8)
- [ ] Test the FastRCNN with ROI Algin (4/10)
- [ ] MaskRCNN using ResNet-50-C4 (4/12)
- [ ] FPN backbone (4/14)
- [ ] Testing all code
- [ ] Turing hyper-parameters
- [ ] Pretrained models
- [ ] Demo code
