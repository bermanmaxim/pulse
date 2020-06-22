from glob import glob
from PIL import Image
from os.path import basename
from tqdm import tqdm

right = 'inp'
left = 'demorand_img'
offset = 100

for f1f, f2 in tqdm(zip(sorted(glob(left + '/*.png')), sorted(glob(right + '/*.png')))):
    f1 = Image.open(f1f)
    f2 = Image.open(f2)
    w, h = f1.size
    f1.paste(f1, (offset, 0))
    f1.paste(f2, (-w//2 + offset, 0))
    f1.save('merged/' + basename(f1f))
