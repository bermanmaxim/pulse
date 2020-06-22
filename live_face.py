from PULSE import PULSE
import cv2
import torch
import argparse
import cv2
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from torchvision.ops import roi_pool
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate


mtcnn = MTCNN(keep_all=True)


kwargs = {}
kwargs['seed'] = 100
kwargs['noise_type'] = 'trainable'
kwargs['opt_name'] = 'adam'
kwargs['learning_rate'] = 0.4
kwargs['lr_schedule'] = 'linear1cycledrop'
kwargs['steps'] = 100
kwargs['loss_str'] = "100*L2+0.05*GEOCROSS"
kwargs['eps'] = 2e-3
kwargs['num_trainable_noise_layers'] = 5
kwargs['tile_latent'] = False
kwargs['bad_noise_layers'] = '17'
kwargs['save_intermediate'] = False

FACE_RESOLUTION = 256
SUB_RESOLUTION = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pulse_model = PULSE(cache_dir="cache", weights="cache/synthesis.pt")
# model = DataParallel(model)


# cap = cv2.VideoCapture(0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def replace(np_image):
    torch_image = torch.from_numpy(np_image).float().unsqueeze(0)
    torch_image = torch_image.to(device)
    boxes, _ = mtcnn.detect(torch_image)
    if boxes is None: return torch_image
    sizes = (boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    sizes = 1.50 * np.maximum(*sizes)
    sizes = (sizes, sizes)
    centers = ((boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2)
    np_boxes = np.vstack(
        (centers[0] - sizes[0] / 2, centers[1] - sizes[1] / 2, centers[0] + sizes[0] / 2, centers[1] + sizes[1] / 2)).T
    np_boxes = np_boxes.round()

    boxes = torch.from_numpy(np_boxes).to(device)
    torch_image = torch_image.div(255.0).permute(0, 3, 1, 2)
    np_boxes = np_boxes.astype(int)

    faces = roi_pool(torch_image, [boxes], FACE_RESOLUTION)

    for face, box in zip(faces, np_boxes):
        face = interpolate(face.unsqueeze(0), SUB_RESOLUTION)
        # face = torch.from_numpy(np.array(Image.open('oup/test_0.png'))).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
        for j, (HR, LR) in enumerate(pulse_model(face, **kwargs)):
            interp = interpolate(HR, (box[3] - box[1], box[2] - box[0]))
            torch_image[:, :, box[1]:box[3], box[0]:box[2]] = interp

    return torch_image.squeeze(0)
#             toPIL(HR[i].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name[i]}.png")


im = Image.open('test.jpg')
transf = replace(np.array(im))
plt.imshow(transf.permute(1, 2, 0).cpu().numpy())
plt.show()

# while True:
#     # Getting out image by webcam
#     _, image = cap.read()
#     # Converting the image to gray scale
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Get faces into webcam's image
#     detected = detect(rgb)
#
#     # For each detected face, find the landmark.
#     for (i, landmarks) in enumerate(detected):
#
#         # Draw on our image, all the finded cordinate points (x,y)
#         for (x, y) in landmarks:
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#
#     # Show the image
#     cv2.imshow("Output", image)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()

