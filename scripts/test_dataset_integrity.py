import sys
from pathlib import Path
sys.path.insert(0, '.')

import cv2
import numpy as np
from utils.dataset_ccpd import parse_ccpd_filename


jpg_dir = Path('data/raw/ccpd/ccpd_base')
jpg_files = sorted(jpg_dir.glob('*.jpg'))


test_indices = [0, 1, 2, 177289, 177290]

out_dir = Path('outputs/visualizations/integrity_test')
out_dir.mkdir(parents=True, exist_ok=True)

print('Test integralności datasetu CCPD')
print('=' * 60)

for idx in test_indices:
    filepath = jpg_files[idx]
    ann = parse_ccpd_filename(filepath.name)
    
    img = cv2.imread(str(filepath))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    x1, y1, x2, y2 = ann['bbox']
    
   
    img_viz = img.copy()
    cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    
    cv2.imwrite(str(out_dir / f'{idx}_full_{ann["plate_text"]}.jpg'), img_viz)
    
 
    crop = img_rgb[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (256, 64))
    cv2.imwrite(str(out_dir / f'{idx}_crop_{ann["plate_text"]}.png'),
                cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
    
    print(f'[{idx}] plate_text={ann["plate_text"]}  bbox={ann["bbox"]}')
    print(f'       plik: {filepath.name[:50]}')

print(f'\nOtwórz: outputs/visualizations/integrity_test/')
print('Dla każdego indeksu są dwa pliki:')
print('  {idx}_full_*.jpg  — cały obraz z zielonym bbox')
print('  {idx}_crop_*.png  — sam wycięty region')
print('\nSprawdź czy zielony bbox otacza tablicę i czy crop zgadza się z nazwą')