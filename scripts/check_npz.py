from pathlib import Path
import numpy as np

npz_dir = Path('data/processed/ccpd/train')
max_idx = 0
problem_files = []
for f in list(npz_dir.glob('*.npz'))[:500]:
    data = np.load(str(f))
    m = int(data['chars'].max())
    if m > max_idx:
        max_idx = m
    if m >= 94:
        problem_files.append((f.name[:40], data['chars'].tolist()))

print(f'Maksymalny indeks: {max_idx}')
print(f'NUM_CHARS:         94')
print(f'Pliki z indeksem >= 94: {len(problem_files)}')
for name, chars in problem_files[:3]:
    print(f'  {name}: {chars}')