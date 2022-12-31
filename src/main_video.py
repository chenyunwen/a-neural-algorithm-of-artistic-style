import time
import model
import utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import cv2
from settings import SIZE, DEVICE, EPOCHS, STYLE_PATH, CONTENT_PATH_VIDEO, OUTPUT_PATH_VIDEO, STYLE_WEIGHT, CONTENT_WEIGHT #, IMAGE_OR_VIDEO
from utils import imloadFrame, imsaveframe

# # Image Transforms
# loader = T.Compose([
#   T.Resize(SIZE),
#   T.CenterCrop(SIZE),
#   T.ToTensor()
# ])

# for video

style_image = utils.load_image(STYLE_PATH)
input_video = cv2.VideoCapture(CONTENT_PATH_VIDEO)
# Check if video opened successfully
print("Check if video opened successfully")
if not input_video.isOpened():
	print("Error opening video stream or file")

else:
	framecount = 0
	frame_width = int(input_video.get(3))
	frame_height = int(input_video.get(4))
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(OUTPUT_PATH_VIDEO, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (SIZE,SIZE))
	# print("out!, frame_width, frame_height: ", frame_width, frame_height)
	
	# Load Pretrained VGG and Normalization Tensors
	cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
	cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
	cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
	
	start = time.time()

	while input_video.isOpened():
		ret, frame = input_video.read()
		if ret:
			# print("ret")
			framecount += 1
			tensor = imloadFrame(frame).to(DEVICE)
			# tensor = loader(tensor)
			# print("!!tensor: ", type(tensor))
			with torch.no_grad():
				# Bootstrap our target image
				target_image = tensor.clone()

				# Build the model
				print("Build the model")
				# cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
				# model = nn.clear_session()
				my_model, style_losses, content_losses = model.style_cnn(cnn, DEVICE, 
					cnn_normalization_mean, cnn_normalization_std, style_image, tensor)

				# Optimization algorithm
				print("Optimization algorithm")
				optimizer = optim.LBFGS([target_image.requires_grad_()])

				# Run style transfer
				print('Starting the style transfer...')
				run = [0]
				while run[0] < EPOCHS:
					# Closure function is needed for LBFGS algorithm
					def closure():
						# Keep target values between 0 and 1
						target_image.data.clamp_(0, 1)

						optimizer.zero_grad()
						my_model(target_image)
						# torch.save(model.state_dict(), "../models/transform_network.pth")
						style_score = 0
						content_score = 0

						for s1 in style_losses:
							style_score += s1.loss
						for c1 in content_losses:
							content_score += c1.loss

						style_score *= STYLE_WEIGHT
						content_score *= CONTENT_WEIGHT

						loss = style_score + content_score
						loss.backward()

						run[0] += 1
						if run[0] % 10 == 0:
							print('Run: {}'.format(run))
							print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
							print()

						return style_score + content_score

					optimizer.step(closure)
				# torch.save(model.state_dict(), "../models/transform_network.pth")
				target_image.data.clamp_(0, 1)


				if(OUTPUT_PATH_VIDEO is not None):
					# print("OUTPUT_PATH_VIDEO is not None")
					output_frame = imsaveframe(target_image, out)
					# utils.save_image(target_image, '../output/test_temp.png')
				else:
					output_frame = imsaveframe(target_image)
				cv2.imshow('Press Q to exit', output_frame)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
		else:
			break

	stop = time.time()
	print(f"Training time: {stop - start}s")
	print("framecount: ", framecount)

	input_video.release()
	if OUTPUT_PATH_VIDEO is not None:
		out.release()
		# print("out.release()")
	cv2.destroyAllWindows()
