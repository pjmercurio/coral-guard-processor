import json
import cv2
import numpy as np
import imageio.v3 as iio
from pathlib import Path

full_dir  = Path('/Users/paulmercurio/Desktop/Labelme')
crop_dir  = Path('/Users/paulmercurio/Desktop/Labelme_crops')

for json_path in sorted(full_dir.glob('*.json')):
    crop_img_path = crop_dir / (json_path.stem + '.png')
    full_img_path = full_dir / (json_path.stem + '.png')

    if not crop_img_path.exists() or not full_img_path.exists():
        print(f'Skipping {json_path.name} — missing image')
        continue

    # Load both images as uint8 grayscale for template matching
    full_img = cv2.cvtColor(iio.imread(str(full_img_path)), cv2.COLOR_RGB2GRAY)
    crop_img = cv2.cvtColor(iio.imread(str(crop_img_path)), cv2.COLOR_RGB2GRAY)

    # Downscale for speed — then scale offset back up
    scale = 4
    full_s = cv2.resize(full_img, (full_img.shape[1]//scale, full_img.shape[0]//scale))
    crop_s = cv2.resize(crop_img, (crop_img.shape[1]//scale, crop_img.shape[0]//scale))

    result = cv2.matchTemplate(full_s, crop_s, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x_offset = max_loc[0] * scale
    y_offset = max_loc[1] * scale
    print(f'{json_path.name}: crop offset = ({x_offset}, {y_offset})')

    # Load and transform the labelme JSON
    with open(json_path) as f:
        label_data = json.load(f)

    # Shift all polygon points into crop space
    for shape in label_data['shapes']:
        shape['points'] = [
            [p[0] - x_offset, p[1] - y_offset]
            for p in shape['points']
        ]

    # Update image path and dimensions to match the crop
    crop_h, crop_w = crop_img.shape[:2]
    label_data['imagePath']   = json_path.stem + '.png'
    label_data['imageHeight'] = crop_h
    label_data['imageWidth']  = crop_w
    label_data['imageData']   = None   # labelme will load from file

    out_path = crop_dir / json_path.name
    with open(out_path, 'w') as f:
        json.dump(label_data, f, indent=2)
    print(f'  → Saved {out_path}')