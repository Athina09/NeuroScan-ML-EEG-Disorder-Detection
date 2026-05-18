
import os, numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BONN_DIR = 'data/bonn'
SAVE_DIR = 'data/processed'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

def bandpass(signal, lo=0.5, hi=45.0, fs=173, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, signal)

def preprocess(signal, fs=173):
    try: signal = bandpass(signal, fs=fs)
    except: pass
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def segment(signal, seg_len=512, overlap=0.5):
    step = int(seg_len * (1 - overlap))
    return np.array([signal[i:i+seg_len] for i in range(0, len(signal)-seg_len+1, step)])

print('Loading BONN...')
folders = {'Z':0,'O':1,'N':2,'F':3,'S':4}
X, y = [], []
for folder, label in folders.items():
    path = os.path.join(BONN_DIR, folder)
    if not os.path.exists(path): print(f'Missing {path}'); continue
    files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
    print(f'  {folder}: {len(files)} files')
    for fname in files:
        try:
            sig = np.loadtxt(os.path.join(path, fname))
            sig = preprocess(sig)
            segs = segment(sig)
            X.extend(segs); y.extend([label]*len(segs))
        except: pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
y2 = (y == 4).astype(np.int64)
np.save(f'{SAVE_DIR}/bonn_X.npy', X)
np.save(f'{SAVE_DIR}/bonn_y5.npy', y)
np.save(f'{SAVE_DIR}/bonn_y2.npy', y2)
print(f'Saved {len(X)} segments')
print(f'Seizure: {np.sum(y2)}, Normal: {np.sum(1-y2)}')

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for cls, color, title in [(0,'green','Healthy (Z)'), (4,'red','Seizure (S)')]:
    idx = np.where(y == cls)[0]
    ax = axes[0] if cls==0 else axes[1]
    if len(idx): ax.plot(X[idx[0]], color=color, linewidth=0.8)
    ax.set_title(title); ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('results/sample_signals.png', dpi=150)
print('Saved: results/sample_signals.png')
print('DONE! Now run: python3 06_train_and_save_models.py')
