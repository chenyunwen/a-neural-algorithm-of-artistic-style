import torch

DEVICE = torch.device('cuda')
SIZE = 256      # 512
EPOCHS = 500    # 300
# STYLE_PATH = '../input/Under-the-Wave-off-Kanagawa.jpg'
# STYLE_PATH = '../input/the-muse.jpg'
# STYLE_PATH = '../input/chicken.jpg'
# STYLE_PATH = '../input/monet.jpg'
# STYLE_PATH = '../input/starry-night.jpg'
STYLE_PATH = '../input/Arles.jpg'

# STYLE_PATH = '../input/other_12.jpg'


STYLE_WEIGHT = 1000000
# CONTENT_PATH = '../input/lenna.jpg'
# CONTENT_PATH = '../input/chicken.jpg'
CONTENT_PATH = '../input/image_1.jpg'
CONTENT_PATH_VIDEO = '../input/video_1.mp4'

CONTENT_WEIGHT = 1
# OUTPUT_PATH = '../output/Gatys_chicken_S256_E500_lenna2.png'
OUTPUT_PATH = '../output/Gatys_Arles_S256_E500_image_1.png'
OUTPUT_PATH_VIDEO = '../output/Gatys_monet_S256_E500_video_0.mp4'

# IMAGE_OR_VIDEO = 0  # 0 for image, 1 for video