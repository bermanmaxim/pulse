from PULSE import PULSE
import torch
import dlib
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
from live_predict import align_face


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


kwargs = {}
kwargs['seed'] = 100
kwargs['noise_type'] = 'trainable'
kwargs['opt_name'] = 'adam'
kwargs['learning_rate'] = 0.4
kwargs['lr_schedule'] = 'linear1cycledrop'
kwargs['steps'] = 100
kwargs['loss_str'] = "100*L2+0.05*GEOCROSS"
kwargs['eps'] = 4e-3
kwargs['num_trainable_noise_layers'] = 5
kwargs['tile_latent'] = False
kwargs['bad_noise_layers'] = '17'
kwargs['save_intermediate'] = False

FACE_RESOLUTION = 256
SUB_RESOLUTION = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pulse_model = PULSE(cache_dir="cache", weights="cache/synthesis.pt", reinit=True)
# model = DataParallel(model)


cap = cv2.VideoCapture(0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def replace(pil_image):
    np_image = np.array(pil_image).astype(np.float32) / 255
    faces = align_face(pil_image, predictor)
    torch_faces = []
    for face, quad in faces:
        face = torch.from_numpy(np.array(face)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        torch_faces.append((face, quad))

    for face, quad in torch_faces:
        face = interpolate(face, SUB_RESOLUTION)
        for j, (HR, LR) in enumerate(pulse_model(face, **kwargs)):
            break
        H, W = HR.size(2), HR.size(3)
        HR = HR.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pts1 = np.float32([[0, 0], [0, H], [W, H], [W, 0]])
        pts2 = quad.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        np_image = cv2.warpPerspective(HR, matrix, (np_image.shape[1], np_image.shape[0]), np_image,
                                       borderMode=cv2.BORDER_TRANSPARENT)

    return np_image


# im = Image.open('test.jpg')
# transf = replace(im)
# plt.imshow(transf)
# plt.show()

while True:
    # Getting out image by webcam
    _, image = cap.read()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = replace(Image.fromarray(rgb))

    # Show the image
    cv2.imshow("Output", cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

