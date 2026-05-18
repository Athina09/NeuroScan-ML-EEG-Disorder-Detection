
import os, json, time, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SAVE_DIR = 'data/processed'
MODEL_DIR = 'saved_models'
RESULT_DIR = 'results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

class CNNModel(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,num_classes))
    def forward(self, x):
        x = x.unsqueeze(1); x = self.features(x); x = x.mean(dim=2); return self.classifier(x)

class BiLSTMModel(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(1,128,batch_first=True,bidirectional=True,dropout=0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.lstm2 = nn.LSTM(256,64,batch_first=True,bidirectional=True,dropout=0.2)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,num_classes))
    def forward(self, x):
        x = x.unsqueeze(2); x,_ = self.lstm1(x)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x,_ = self.lstm2(x); x = x[:,-1,:]
        x = self.bn2(x); return self.classifier(x)

class HybridModel(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=128,num_heads=4,dropout=0.1,batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.bilstm = nn.LSTM(128,64,batch_first=True,bidirectional=True,dropout=0.2)
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,num_classes))
    def forward(self, x):
        x = x.unsqueeze(1); x = self.cnn(x); x = x.permute(0,2,1)
        a,_ = self.attention(x,x,x); x = self.norm(x+a)
        x,_ = self.bilstm(x); x = x[:,-1,:]; x = self.bn(x); return self.classifier(x)

def train_task(X, y, num_classes, task_key, class_names, label_names):
    print(f'\nTraining: {task_key} | {len(X)} samples | {num_classes} classes')
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    X_train,X_val,y_train,y_val = train_test_split(Xt,yt,test_size=0.2,random_state=42,stratify=yt)
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    train_dl = DataLoader(TensorDataset(X_train,y_train), batch_size=32, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,y_val), batch_size=64)
    results = {}
    for mkey, ModelClass in [('cnn',CNNModel),('bilstm',BiLSTMModel),('hybrid',HybridModel)]:
        print(f'  -- {mkey.upper()} --')
        model = ModelClass(X.shape[1], num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        best_acc, best_state, no_improve = 0, None, 0
        history = {'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[]}
        for epoch in range(60):
            model.train()
            tl, tc, tt = 0, 0, 0
            for xb,yb in train_dl:
                xb,yb = xb.to(DEVICE),yb.to(DEVICE)
                optimizer.zero_grad(); out=model(xb); loss=criterion(out,yb)
                loss.backward(); optimizer.step()
                tl+=loss.item(); tc+=(out.argmax(1)==yb).sum().item(); tt+=len(yb)
            model.eval(); vl,vc,vt=0,0,0
            with torch.no_grad():
                for xb,yb in val_dl:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); out=model(xb)
                    vl+=criterion(out,yb).item(); vc+=(out.argmax(1)==yb).sum().item(); vt+=len(yb)
            ta,va=tc/tt,vc/vt; tl/=len(train_dl); vl/=len(val_dl)
            history['train_acc'].append(round(ta,4)); history['val_acc'].append(round(va,4))
            history['train_loss'].append(round(tl,4)); history['val_loss'].append(round(vl,4))
            scheduler.step(vl)
            if va > best_acc: best_acc=va; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
            else: no_improve+=1
            if (epoch+1)%10==0: print(f'    Epoch {epoch+1}: train={ta:.3f} val={va:.3f}')
            if no_improve>=10: print(f'    Early stop at {epoch+1}'); break
        model.load_state_dict(best_state)
        model.eval(); preds,labs=[],[]
        with torch.no_grad():
            for xb,yb in val_dl:
                out=model(xb.to(DEVICE)); preds.extend(out.argmax(1).cpu().numpy()); labs.extend(yb.numpy())
        acc=accuracy_score(labs,preds); prec=precision_score(labs,preds,average='weighted',zero_division=0)
        rec=recall_score(labs,preds,average='weighted',zero_division=0); f1=f1_score(labs,preds,average='weighted',zero_division=0)
        print(f'    FINAL -> Acc:{acc*100:.2f}% Prec:{prec*100:.2f}% Rec:{rec*100:.2f}% F1:{f1*100:.2f}%')
        torch.save({'model_state':best_state,'model_class':mkey,'input_len':X.shape[1],'num_classes':num_classes,'class_names':class_names,'label_names':label_names}, f'{MODEL_DIR}/{mkey}_{task_key}.pt')
        with open(f'{RESULT_DIR}/{mkey}_{task_key}_history.json','w') as f: json.dump(history,f)
        results[mkey] = {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}
    meta = {'task_key':task_key,'class_names':class_names,'label_names':label_names,'num_classes':num_classes,'input_len':int(X.shape[1]),'results':{k:{m:float(v) for m,v in r.items()} for k,r in results.items()}}
    with open(f'{MODEL_DIR}/{task_key}_meta.json','w') as f: json.dump(meta,f,indent=2)
    return results

t0 = time.time()
all_results = {}
if os.path.exists(f'{SAVE_DIR}/bonn_X.npy'):
    X=np.load(f'{SAVE_DIR}/bonn_X.npy'); y2=np.load(f'{SAVE_DIR}/bonn_y2.npy'); y5=np.load(f'{SAVE_DIR}/bonn_y5.npy')
    all_results['bonn_binary'] = train_task(X,y2,2,'bonn_binary',['Normal','Seizure'],{0:'Normal',1:'Seizure Detected'})
    all_results['bonn_5class'] = train_task(X,y5,5,'bonn_5class',['Healthy-Z','Healthy-O','Interictal-N','Interictal-F','Ictal/Seizure'],{0:'Healthy(Z)',1:'Healthy(O)',2:'Interictal-N',3:'Interictal-F',4:'Ictal/Seizure'})

print(f'\n=== DONE in {(time.time()-t0)/60:.0f} mins ===')
for task,res in all_results.items():
    print(f'\n{task}:')
    for m,r in res.items(): print(f'  {m.upper():8s}: {r["accuracy"]*100:.2f}%')
print('\nNow run: streamlit run app.py')
