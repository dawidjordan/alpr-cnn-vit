import sys
sys.path.insert(0, '.')
from utils.dataset_ccpd import parse_ccpd_filename, PROVINCES, ALPHABETS, DIGITS

filename = '00268199233717-92_90-316&515_409&560-413&558_317&553_318&516_414&521-0_0_26_17_24_32_33-148-26.jpg'
parts = filename.split('-')
raw = [int(x) for x in parts[4].split('_')]
print(f'Surowe indeksy z nazwy: {raw}')
print(f'raw[0]={raw[0]} -> PROVINCES[{raw[0]}]={PROVINCES[raw[0]]}')
print(f'raw[1]={raw[1]} -> ALPHABETS[{raw[1]}]={ALPHABETS[raw[1]]}')
print(f'raw[2]={raw[2]} -> DIGITS[{raw[2]}]={DIGITS[raw[2]]}')

ann = parse_ccpd_filename(filename)
print(f'plate_chars (globalne): {ann["plate_chars"]}')
print(f'plate_text: {ann["plate_text"]}')
print(f'Max indeks: {max(ann["plate_chars"])}')
print(f'NUM_CHARS:  94')
print(f'Problem:    {max(ann["plate_chars"]) >= 94}')