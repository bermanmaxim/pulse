from PULSE import PULSE
import torch
import dlib
from PIL import Image
import numpy as np
import cv2
from torch.nn.functional import interpolate
from live_align import align_face
import argparse
import os
from glob import glob


parser = argparse.ArgumentParser()

parser.add_argument('--input-video', type=str, default='testvid.mp4', help='directory with unprocessed images')
parser.add_argument('--output-video', type=str, default='outvid.mp4', help='output video')
parser.add_argument('--down-size', type=int, default=32, help='size to downscale the input images to, must be power of 2')
parser.add_argument('--seed', type=int, default=100, help='manual seed to use')
parser.add_argument('--framerate', type=int, default=30, help='output framerate')
parser.add_argument('--eps', type=float, default=0.004, help='eps for termination criterion')
parser.add_argument('--reuse-latent', action='store_true', help='reuse latent between frames')
parser.add_argument('--resume', action='store_true', help='resume aborted computation')

args = parser.parse_args()

tmpdir_out = os.path.splitext(args.output_video)[0] + '_img'

os.makedirs(tmpdir_out, exist_ok=True)

if os.path.splitext(args.input_video)[1] == '':
    tmpdir_in = args.input_video
else:
    print("extracting video frames...")
    tmpdir_in = os.path.splitext(args.input_video)[0] + '_img'
    os.makedirs(tmpdir_in, exist_ok=True)
    os.system(f"ffmpeg -i {args.input_video} '{tmpdir_in}/out%05d.png'")


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


kwargs = {}
kwargs['seed'] = args.seed
kwargs['noise_type'] = 'trainable'
kwargs['opt_name'] = 'adam'
kwargs['learning_rate'] = 0.4
kwargs['lr_schedule'] = 'linear1cycledrop'
kwargs['steps'] = 100
kwargs['loss_str'] = "100*L2+0.05*GEOCROSS"
kwargs['eps'] = args.eps
kwargs['num_trainable_noise_layers'] = 5
kwargs['tile_latent'] = False
kwargs['bad_noise_layers'] = '17'
kwargs['save_intermediate'] = False

SUB_RESOLUTION = args.down_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pulse_model = PULSE(cache_dir="cache", weights="cache/synthesis.pt", reinit=args.reuse_latent)
# model = DataParallel(model)


vidcap = cv2.VideoCapture('testvid.mp4')
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


todo = sorted(glob(tmpdir_in + '/*.png'))
for i, fname in enumerate(todo):
    dest = tmpdir_out + '/' + os.path.basename(fname)
    if args.resume and os.path.isfile(dest):
        continue
    print(f"{i}/{len(todo)}", end=" ")
    transformed = replace(Image.open(fname))
    transformed = Image.fromarray((transformed * 255).astype(np.uint8))
    transformed.save(dest)


os.system(f"ffmpeg -framerate {args.framerate} -pattern_type glob -i '{tmpdir_out}/*.png' "
          f"-c:v libx264 -pix_fmt yuv420p {args.output_video}")