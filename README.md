# A Neural Algorithm of Artistic Style

## What we did?
1. Try to use it on video style transfer.
2. Calculate execution time.

## How to use?
Change the parameter in `src/settings.py`.
- change input style image:
    in `src/settings.py`, and change `STYLE_PATH = '../path/to/style'`
- chaneg input content:
    in `src/settings.py`, and change `CONTENT_PATH = '../path/to/content'`

Run this code:
- For image style transfer:
```
python src/main.py
```
- For video style transfer:
```
python src/main_video.py
```

## Our result

**The following content comes from the original Repo.**

Source: https://arxiv.org/pdf/1508.06576.pdf  
Authors: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

This is a pytorch implementation of neural style transfer as described in the above paper. It works by extracting the content and style of different images by feeding them through a convolutional neural network and looking at the features at different layers in the network. We then perform gradient descent on a target image and try to minimize the loss between that target image and both the content and style features. The result is a combination of the two.

![Combined](output/combined.png)

## Requirements + Versions

- [Python 3.6](https://www.python.org/)
- [PyTorch 1.0](https://pytorch.org/)
- [Pillow 5.3](https://pillow.readthedocs.io/en/5.3.x/)
- [Matplotlib 3.0](https://matplotlib.org/)

## Running

All the config is done within the main file. Tweak the variables from within.

```
$ python3 src/main.py
```

Results will be saved in `/output`