import sys
from pathlib import Path
sys.path.insert(0, '.')
from collections import defaultdict

image_dir = Path('data/raw/compcars/image')
make_counts = defaultdict(int)

for make_dir in sorted(image_dir.iterdir()):
    if not make_dir.is_dir():
        continue
    count = sum(1 for _ in make_dir.rglob('*.jpg'))
    make_counts[make_dir.name] = count


total_makes = len(make_counts)
total_images = sum(make_counts.values())
print(f'Łącznie marek: {total_makes}')
print(f'Łącznie obrazów: {total_images:,}')
print()


for min_samples in [50, 100, 200, 300, 500, 1000]:
    filtered = {k: v for k, v in make_counts.items() if v >= min_samples}
    covered = sum(filtered.values())
    print(f'Min {min_samples:>4} zdjęć → {len(filtered):>3} marek | '
          f'{covered:>6,} obrazów ({covered/total_images*100:.1f}% datasetu)')

print()
print('Top 10 marek wg liczby zdjęć:')
for make, count in sorted(make_counts.items(), key=lambda x: -x[1])[:10]:
    print(f'  make_id={make:<6} {count:>5,} zdjęć')

print()
print('Bottom 10 marek wg liczby zdjęć:')
for make, count in sorted(make_counts.items(), key=lambda x: x[1])[:10]:
    print(f'  make_id={make:<6} {count:>5,} zdjęć')