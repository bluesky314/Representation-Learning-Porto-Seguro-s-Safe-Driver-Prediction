# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
save_initial='Embed_small'

bs=128
init_lr=0.001
scheduler_step_size = 3

save_every=50# save every _ epochs

epochs_run=1000 # No of epochs

from tensorboardX import SummaryWriter
writer = SummaryWriter()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns, numpy as np
from scipy.stats import norm
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.over_sampling import SMOTE

print(os.getcwd())
df = pd.read_csv('/home/Drive/rahul/PortoSeguroSafeDriverPrediction/train.csv')
df.drop(['id'],axis=1,inplace=True)
print(df.target[:10])
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def create_meta(df):    
    data = []
    for f in df.columns:
        # Defining the role
        if f == 'target':
            continue
        elif f == 'id':
            role = 'id'
        else:
            role = 'input'

        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif df[f].dtype == float:
            level = 'interval'
        elif df[f].dtype == int:
            level = 'ordinal'

        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False

        # Defining the data type 
        dtype = df[f].dtype

        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)

    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    return(meta)
# Any results you write to the current directory are saved as output.
meta=create_meta(df)
# meta.drop(['target'],inplace=True)
meta.drop(['keep'],axis=1,inplace=True)
meta.drop(['role'],axis=1,inplace=True)


# Popular target encoding with noise, used in many good kernals. Target encoding will help not doing one-hot and increasing feature numbers
def gini_xgb(preds, dtrain):    # This can be used in xgboost as metric
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
#     assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
#     ft_tst_series = pd.merge(
#         tst_series.to_frame(tst_series.name),
#         averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#         on=tst_series.name,
#         how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
#     # pd.merge does not keep the index so restore it
#     ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level) #, add_noise(ft_tst_series, noise_level)


# Feature Engen

df = pd.read_csv('/home/Drive/rahul/PortoSeguroSafeDriverPrediction/train.csv')
df.drop(['id'],axis=1,inplace=True)
orig=df.copy()



df=df.iloc[:100000,:]



# removing calc cols increases baseline xgboost from 0.22 to 0.23227886570
for e in df.columns:
    if 'calc' in e: df.drop(e,axis=1,inplace=True)

meta=create_meta(df)
# meta.drop(['target'],inplace=True)
meta.drop(['keep'],axis=1,inplace=True)
meta.drop(['role'],axis=1,inplace=True)

np.shape(df)

np.shape(df[meta[meta.level=='binary'].index].describe()) # 17 original binary 



np.sum(df.target==1)

# Handle missing vals first
# can impute/create feat of binary missing or not
# it doesnt make sense to do mode imputation is a class does not have large majority, creat feat here
# continous vars - mean/median

N=np.shape(df)[0]
N

#missing continous
for e in meta[meta.level=='interval'].index :  
    if np.min(df[e])==-1:
        print(e, 'has', np.sum(df[e]==-1),'missing vals which is ',np.sum(df[e]==-1)/N ,'from 1' ) # consdier log transform for 2
        df[e][df[e]==-1]=np.mean(df[e])

#missing Ordinal
# Can we just assign ordinal to highest? Maybe 2 would be better....
# only ps_car_11 which does have a class majority at 3
print(df['ps_car_11'].value_counts())
df['ps_car_11'][df['ps_car_11']==-1]=3



#missing categorical
from scipy import stats

mode_r=["ps_ind_02_cat","ps_ind_04_cat","ps_ind_05_cat","ps_car_02_cat","ps_car_07_cat","ps_car_09_cat",'ps_car_11']
for e in mode_r:
    print(e, 'has', np.sum(df[e]==-1),'missing vals which is ',100*np.sum(df[e]==-1)/N ,'%' )
    df[e][df[e]==-1]=stats.mode(df[e])[0]

# Drop too many missing vars
df.drop(["ps_car_03_cat","ps_car_05_cat"],axis=1,inplace=True)

# df.drop(['ps_car_11_cat'],axis=1,inplace=True) # drop this as it has wayyy to many categories 104

#outliers
# We can clip outliers, or create a feat to indicate outlier presence, or impute vals

def IQR_outlier(df,e):
    quartile_1,quartile_3 = np.percentile(df[e],[25,75])
    IQR=quartile_3-quartile_1
    upper_lim=quartile_3+1.5*IQR
    lower_lim=quartile_1-1.5*IQR

    print(e, ' has',np.shape(df[e][df[e]>upper_lim])[0],' outliers which is ',100*np.shape(df[e][df[e]>upper_lim])[0]/N ,'%' )
    df['ps_reg_02'][df['ps_reg_02']>upper_lim]=upper_lim
    
    return(df)

outlier_list=['ps_reg_02',"ps_reg_03"
,"ps_car_12"
,"ps_car_13"
]
for e in outlier_list:
    df=IQR_outlier(df,e)#36793

meta=create_meta(df)

# ------One hot Encode Branch----- 

v = meta[meta.level=='nominal'].index
print('Before dummification we have {} variables in train'.format(df.shape[1]))
train = pd.get_dummies(df, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))

# 104 of these are from ps_car_11_cat, we can remove that col or something else

train.head()

meta=create_meta(train) # we shifted from df to train



# train.drop(['target'],1,inplace=True) #<- REMOVE TARGET
meta=create_meta(train) 
binary_cols = np.concatenate([ meta[meta.level=='binary'].index,meta[meta.level=='nominal'].index ])
# train[binary_cols]
# np.shape(binary_cols)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train[meta[meta.level=='interval'].index] = scaler.fit_transform(train[meta[meta.level=='interval'].index])


# Embedding Network 

print(np.shape(train)),train.target.value_counts()[0]

# Upsample

from sklearn.utils import resample
print(train.target.value_counts())
df_majority = train[train.target==0]
df_minority = train[train.target==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=train.target.value_counts()[0]//2,    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.target.value_counts()


train=df_upsampled
y=train.target
train=train.drop(['target'],axis=1)
print(np.shape(train))


ind_cols=[]
reg_cols=[]
car_cols=[]
for e in train.columns:
    if 'ind' in e:
        ind_cols.append(e)
    elif 'reg' in e:
        reg_cols.append(e)
    elif 'car' in e:
        car_cols.append(e)

np.shape(ind_cols),np.shape(reg_cols),np.shape(car_cols)

reg_cols

ind_cols_cat=[]
ind_cols_num=[]
for f in ind_cols:
    if 'bin' in f or f == 'target':
        ind_cols_cat.append(f)
    elif 'cat' in f or f == 'id':
        ind_cols_cat.append(f)
    elif df[f].dtype == float:
        ind_cols_num.append(f)
    elif df[f].dtype == int:
        ind_cols_num.append(f)
        
        
reg_cols_cat=[]
reg_cols_num=[]
for f in reg_cols:
    if 'bin' in f or f == 'target':
        reg_cols_cat.append(f)
    elif 'cat' in f or f == 'id':
        reg_cols_cat.append(f)
    elif df[f].dtype == float:
        reg_cols_num.append(f)
    elif df[f].dtype == int:
        reg_cols_num.append(f)
        
        
car_cols_cat=[]
car_cols_num=[]
for f in car_cols:
    if 'bin' in f or f == 'target':
        car_cols_cat.append(f)
    elif 'cat' in f or f == 'id':
        car_cols_cat.append(f)
    elif df[f].dtype == float:
        car_cols_num.append(f)
    elif df[f].dtype == int:
        car_cols_num.append(f)

        

all=[np.array(ind_cols_cat),np.array(ind_cols_num),np.array(reg_cols_cat),np.array(reg_cols_num),np.array(car_cols_cat),
     np.array(car_cols_num)]
for e in all:
    print(np.shape(e))

names_list=[str("ind_cols_cat"),str("ind_cols_num"),str("reg_cols_cat"),str("reg_cols_num"),str("car_cols_cat"),
     str("car_cols_num")]
names_list

dictdf={}
for i in range(len(all)):
    name = names_list[i] +'_df'
    
    dictdf[name]=train[all[i]]
    

dictdf.keys()

ind_cols_cat_df=dictdf["ind_cols_cat_df"]
ind_cols_num_df=dictdf["ind_cols_num_df"]
reg_cols_num_df=dictdf["reg_cols_num_df"]
car_cols_cat_df=dictdf["car_cols_cat_df"]
car_cols_num_df=dictdf["car_cols_num_df"]

for e in dictdf.keys():
    print(e,np.shape(dictdf[e]))
#     print(dictdf[e].iloc[10])

# as is empty
del dictdf['reg_cols_cat_df']


from torch.utils.data import Dataset, DataLoader

class EDataset(Dataset):
    """Returns the different components needed for EmbeddingNet seperately as a list"""

    def __init__(self, ind_cols_cat_df, ind_cols_num_df, reg_cols_num_df, car_cols_cat_df, car_cols_num_df,y,type="train",pct=0.85): # 

        
        N=np.shape(ind_cols_cat_df)[0]
        train_pct=int(pct*N)
        val_pct=int((1-pct)*N)
#         print(train_pct,val_pct)
        if type=="train":
            self.ind_cols_cat_d=ind_cols_cat_df.iloc[:train_pct,:]
            self.ind_cols_num_d=ind_cols_num_df.iloc[:train_pct,:]
            self.reg_cols_num_d=reg_cols_num_df.iloc[:train_pct,:]
            self.car_cols_cat_d=car_cols_cat_df.iloc[:train_pct,:]
            self.car_cols_num_d=car_cols_num_df.iloc[:train_pct,:]
            self.y=y[:train_pct]
            
            
        else:
            self.ind_cols_cat_d=ind_cols_cat_df.iloc[-val_pct:,:]
            self.ind_cols_num_d=ind_cols_num_df.iloc[-val_pct:,:]
            self.reg_cols_num_d=reg_cols_num_df.iloc[-val_pct:,:]
            self.car_cols_cat_d=car_cols_cat_df.iloc[-val_pct:,:]
            self.car_cols_num_d=car_cols_num_df.iloc[-val_pct:,:]
            self.y=y[-val_pct:]
            
    def __len__(self):
        return np.shape(self.ind_cols_cat_d)[0]

    def __getitem__(self, idx):
        ind_cols_cat_d=torch.tensor(self.ind_cols_cat_d.iloc[idx])
        ind_cols_num_d=torch.tensor(self.ind_cols_num_d.iloc[idx])
        reg_cols_num_d=torch.tensor(self.reg_cols_num_d.iloc[idx])
        car_cols_cat_d=torch.tensor(self.car_cols_cat_d.iloc[idx])
        car_cols_num_d=torch.tensor(self.car_cols_num_d.iloc[idx])
#         print(self.y[:10])
        target=torch.tensor(self.y.iloc[idx])
        
        sample = [ind_cols_cat_d,ind_cols_num_d,reg_cols_num_d,car_cols_cat_d,car_cols_num_d,target]
        return sample


import torch.nn.functional as F

class Embedding_Net(torch.nn.Module):

    def __init__(self,df):
        super(Embedding_Net, self).__init__()
#         self.embeddings_ind = torch.nn.Embedding(np.shape(df['ind_cols_cat_df'])[1], 3)
#         self.embeddings_car = torch.nn.Embedding(np.shape(df['car_cols_cat_df'])[1], 50)
        
        embds=[12,100]
        self.embeddings_ind = torch.nn.Linear(np.shape(df['ind_cols_cat_df'])[1], embds[0])
        self.embeddings_car = torch.nn.Linear(np.shape(df['car_cols_cat_df'])[1], embds[1])
        
        
        linear_shapes=[32,128,32] #ind,car,reg
        self.linear_ind = torch.nn.Linear( embds[0] + np.shape(df['ind_cols_num_df'])[1], linear_shapes[0]) # this will take in embeddings plus concatonated numeric values to it
        
        self.linear_car = torch.nn.Linear( embds[1]+np.shape(df['car_cols_num_df'])[1], linear_shapes[1])
        
        self.linear_reg = torch.nn.Linear( np.shape(df['reg_cols_num_df'])[1], linear_shapes[2])
        
        lin=300
        self.linear_all1 = torch.nn.Linear(sum(linear_shapes), lin)
        self.linear_all2 = torch.nn.Linear(lin, 1)
        
        self.d1=torch.nn.Dropout(p=0.4)
        self.d2=torch.nn.Dropout(p=0.4)
        self.d3=torch.nn.Dropout(p=0.4)
        self.d4=torch.nn.Dropout(p=0.4)

    def forward(self, ind_cols_cat_df, ind_cols_num_df, reg_cols_num_df, car_cols_cat_df, car_cols_num_df): # give dictdf as input
        
        embeds_ind = F.tanh(self.embeddings_ind(ind_cols_cat_df))#.view((1, -1))
#         print('1',embeds_ind.size())
        inds=torch.cat((embeds_ind,ind_cols_num_df.float()),dim=1)
        
#         print('2',inds.size())
        embeds_car = F.tanh(self.embeddings_car(car_cols_cat_df))#.view((1, -1))
        cars=torch.cat((embeds_car,car_cols_num_df),dim=1)
        
#         print('3',cars.size())
        l1=self.d1(F.relu(self.linear_ind(inds)))
        l2=self.d2(F.relu(self.linear_car(cars)))
        l3=self.d3(F.relu(self.linear_reg(reg_cols_num_df)))
        
        
        alllinear=torch.cat((l1,l2,l3),dim=1)
#         print('all',alllinear.size())
        alllinear2=self.d4(F.relu(self.linear_all1(alllinear)))
        
        output=self.linear_all2(alllinear2)
        
        return(output)

embednet=Embedding_Net(dictdf).cuda()


bs=2*192

ds=EDataset(ind_cols_cat_df, ind_cols_num_df, reg_cols_num_df, car_cols_cat_df, car_cols_num_df,y)
data_loader_train = DataLoader(ds, bs, shuffle=True, num_workers=2)

ds_val=EDataset(ind_cols_cat_df, ind_cols_num_df, reg_cols_num_df, car_cols_cat_df, car_cols_num_df,y,type='val')
data_loader_val = DataLoader(ds_val, bs, shuffle=True, num_workers=2)


loaders={'train':data_loader_train,'val': data_loader_val}




from torch.optim.lr_scheduler import StepLR

opt_step_size=1
opt_gamma=0.55   

init_lr=0.01
binary_loss = torch.nn.BCELoss()

sigmoid=torch.nn.Sigmoid()

opt = torch.optim.Adam(embednet.parameters(), lr=init_lr)
scheduler = StepLR(opt, step_size=2, gamma=0.99)

for idx, batch_data in enumerate(data_loader_val):
    print(idx)
    for e in batch_data:
        print('ins',np.shape(e))
    
    outs=embednet(batch_data[0].float().cuda(),batch_data[1].float().cuda(),batch_data[2].float().cuda(),batch_data[3].float().cuda(),batch_data[4].float().cuda())
    
    print(outs.size(),'outs')
    break
def r(x): return(round(x,2))

def trainer_embed(model,epochs):
    L=[]
    V=[]
    counter_t=0
    counter_v=0
    for epoch in range(epochs):
        scheduler.step()
        if (epoch+1)%save_every==0:
                        checkpoint = {'train_loss': L, 'valid_loss':V,'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }

                        torch.save(checkpoint, '/home/Drive/rahul/PortoSeguroSafeDriverPrediction/'+save_initial+str(epoch)+'V'+str(r(sum(V)/len(V))) )
                        
                        print('saved----------')
                        
        for e in loaders:
            if e=='train':  model.train() ; grad=True
            else: model.eval() ; grad=False

            for idx, batch_data in enumerate(loaders[e]):
                if e=='train':counter_t+=1
                else:counter_v+=1
                
                target=Variable(batch_data[-1]).float().cuda()
                pred=embednet(batch_data[0].float().cuda(),batch_data[1].float().cuda(),batch_data[2].float().cuda(),batch_data[3].float().cuda(),batch_data[4].float().cuda() )
                pred=sigmoid(pred)
                loss=binary_loss(pred,target) 
                
                
                
                if e=='train':
                    L.append(loss.item())
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                    pred=pred.squeeze(1).detach().cpu().numpy()# for gini
                    pred[pred<0.5]=0 
                    pred[pred>0.5]=1
                    
                    
                    print(e,'Epoch:',epoch,'Batch:',idx,'Loss:',sum(L)/len(L))#,'Gini:',eval_gini(target.detach().cpu().numpy(), pred))
                    writer.add_scalar('Total Train Loss', loss.item(), counter_t)

                    
                    writer.add_scalar('Learning Rate', scheduler.get_lr()[-1], counter_t)
                    
                    
                else:
                    V.append(loss.item())
                    writer.add_scalar('Total Train LossV', loss.item(), counter_v)
                    
                    pred=pred.squeeze(1).detach().cpu().numpy()
                    pred[pred<0.5]=0
                    pred[pred>0.5]=1
                    print(e,'Epoch:',epoch,'Batch:',idx,'Loss:',sum(V)/len(V),'Gini:',eval_gini(target.detach().cpu().numpy(), pred))
    return(model,L,V)

embednet,L,V=trainer_embed(embednet,30)