
bs=128
init_lr=0.001
scheduler_step_size = 3

save_every=50# save every _ epochs

epochs_run=1000 # No of epochs
swap_col_no=183 # how many cols to randomly swap at each swap step,183 is all

save_initial='DAE_first'
import os
gpu_number = "0"            

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
#run here
# deactivate
# cd ..
# cd rahul
# source py3-env/bin/activate
# cd ..
# cd Drive
# cd rahul





from tensorboardX import SummaryWriter
writer = SummaryWriter()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
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
# import seaborn as sns, numpy as np
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

df = pd.read_csv('/home/Drive/rahul/PortoSeguroSafeDriverPrediction/train.csv')
df.drop(['id'],axis=1,inplace=True)

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



df=df.iloc[:300000,:]



# removing calc cols increases baseline xgboost from 0.22 to 0.23227886570
for e in df.columns:
    if 'calc' in e: df.drop(e,axis=1,inplace=True)

meta=create_meta(df)
# meta.drop(['target'],inplace=True)
meta.drop(['keep'],axis=1,inplace=True)
meta.drop(['role'],axis=1,inplace=True)

np.shape(df)

np.shape(df[meta[meta.level=='binary'].index].describe()),np.shape(df[meta[meta.level=='interval'].index].describe()) # 11 original binary 



np.sum(df.target==1)


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

# Make new feat as heavy concentration after 102
# sns.countplot(df["ps_car_11_cat"],palette='summer')
# plt.show()
# print(np.unique(df["ps_car_11_cat"])[:-10]) # last 10

# sns.countplot(df["ps_car_11_cat"][df["ps_car_11_cat"]>100],palette='summer')
# plt.show()
 
# #Make feat =<102 or >102

# df['ps_car_11_cat_bin']=df["ps_car_11_cat"]>102
# df['ps_car_11_cat_bin']=df['ps_car_11_cat_bin'].astype(int)
# print(df.ps_car_11_cat_bin[:10])

# sns.countplot(df["ps_car_11_cat_bin"],palette='summer')
# plt.show()

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

# Handline Ordinal
# ps_calc_05 can bin 5,6

# ps_calc_06 can bin 0-4

# ps_calc_10 outliers > 17

# ps_calc_11 outliers > 13

# ps_calc_12 outliers> 6

# ps_calc_13 outliers > 9

# ps_calc_14 outliers > 16

# removed in calc
# df['ps_calc_05'][df['ps_calc_05']>5]=6
# df['ps_calc_06'][df['ps_calc_06']<4]=4
# df['ps_calc_10'][df['ps_calc_10']>17]=17

# df['ps_calc_11'][df['ps_calc_11']>13]=13

# df['ps_calc_12'][df['ps_calc_12']>6]=6

# df['ps_calc_13'][df['ps_calc_13']>9]=9

# df['ps_calc_14'][df['ps_calc_14']<16]=16



meta=create_meta(df)

# ------One hot Encode Branch----- 

v = meta[meta.level=='nominal'].index
print('Before dummification we have {} variables in train'.format(df.shape[1]))
train = pd.get_dummies(df, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))

# 104 of these are from ps_car_11_cat, we can remove that col or something else

train.head()

meta=create_meta(train) # takes into account one hot



# train.drop(['target'],1,inplace=True) #<- REMOVE TARGET
meta=create_meta(train) 
binary_cols = np.concatenate([ meta[meta.level=='binary'].index,meta[meta.level=='nominal'].index ])
# train[binary_cols]
# np.shape(binary_cols)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train[meta[meta.level=='interval'].index] = scaler.fit_transform(train[meta[meta.level=='interval'].index])


np.shape(train)

np.shape(binary_cols),np.shape(meta[meta.level=='interval'].index),np.shape(meta[meta.level=='ordinal'].index)

# np.max(np.max(binner,1))

# Create a new df with all binary variables on right end

binner=train[binary_cols]
inter=train[meta[meta.level=='interval'].index]
ordi=train[meta[meta.level=='ordinal'].index]

train_order=pd.concat([inter, ordi,binner], axis=1) # will not have target

train_order.head() # inerval - ordinal - binary

np.max(np.max(train_order.iloc[:,-np.shape(binary_cols)[0]:]))


try:train=train.drop(['target'],axis=1)
except:pass
X = train
y = df.target


# Representation learning neural net

for e in train_order.columns: # check target is not present
    if e=='target': print('Target present')

np.shape(train_order),np.shape(train_order)[1]
# train_order.head(3)

temp=train_order.copy()
import random
# for e in temp.columns:
#     index_samples_orig=random.sample(range(0, np.shape(temp)[0]), swap_amt) # rows to replace
#     index_samples_inplace=random.sample(range(0, np.shape(temp)[0]), swap_amt) # rows to be put in place of the replaced
    
    
# #     print(temp[e].loc[index_samples_orig][:10],np.shape(temp[e].loc[index_samples_orig]))
#     temp[e].loc[index_samples_orig]=np.nan # set rows to nan 
# #     print(temp[e].loc[index_samples_orig][:10])
    
    
#     fill = pd.DataFrame(index = index_samples_orig, data= temp[e].loc[index_samples_inplace].tolist(),columns=[e]) # make a list to fill nan cols with target values
#     temp.fillna(fill,inplace=True)
    
    


from torch.utils.data import Dataset, DataLoader
import random as random
class NDataset(Dataset):
    """Regular dataloader"""

    def __init__(self, type="train",csv_file=train_order): # MUST TAKE ORDERED TRAIN
        self.df=csv_file
        N=np.shape(self.df)[0]
        train_pct=int(0.98*N)
        val_pct=int(0.02*N)
        
        if type=="train":
            self.df=self.df.loc[:train_pct,:]
        else:
            self.df=self.df.loc[-val_pct:,:]
            
    def __len__(self):
        return np.shape(self.df)[0]

    def __getitem__(self, idx):
        row=np.array(self.df.loc[idx])
        
        sample = {'x': row, 'y': row}
        return sample
class NDataset_swap(Dataset):
    """Adding gaussian or uniform additive / multiplicative noise is not optimal since features have different scale or a discrete set of values
        So we use "swap noise". We sample from the feature itself with a certain probability "inputSwapNoise".Default of 0.15 means 15% of features replaced by values from another row."""

    def __init__(self, ncols,type="train",csv_file=train_order,split=0.97): # MUST TAKE ORDERED TRAIN
        self.df=csv_file
        
        
        N=np.shape(self.df)[0]
        train_pct=int(split*N)
        val_pct=int( (1-split) *N)
        
        self.inputSwapNoise=0.15
        
        if type=="train":
            self.df=self.df.iloc[:train_pct,:]
            
            
            self.df_copy=self.df.copy()

            
            swap_amt=int(0.15*N) # number of entires to swap
            
            
            
            rand_cols=random.sample(range(0, np.shape(self.df.columns)[0]), ncols) # ** sample ncols cols to shuffle every epoch
            count=0
            for e in self.df.columns[rand_cols]: # for each col the row indexes have to be sampled again as otherwise we would just replace row x with y
#                 print(self.df.isnull().values.any())
                index_samples_orig=random.sample(range(0, np.shape(self.df)[0]), swap_amt) # rows to replace
                index_samples_inplace=random.sample(range(0, np.shape(self.df)[0]), swap_amt) # rows to be put in place of the replaced
                
#                 print('1',np.shape(index_samples_orig))
#                 print('sum',np.sum(np.array(index_samples_orig)!=np.array(index_samples_inplace)))
#                 print(index_samples_orig[:5],index_samples_inplace[:5])
                
                old=self.df[e]
#                 print('o',self.df[e][:10])
                
                new_column = pd.Series(self.df[e].loc[index_samples_inplace].tolist(), name=e, index=index_samples_orig)
                
                self.df.update(new_column)
#                 fill = pd.DataFrame(index = index_samples_orig, data= self.df[e].loc[index_samples_inplace].tolist(),columns=[e]) # make a list to fill nan cols with target values
#                 self.df.fillna(fill,inplace=True)
#                 new=self.df[e]
    
#                 print(swap_amt,'sum',np.sum(old!=new))
#                 print('o',self.df[e][:10])
                count+=1
#                 print(count)
                

            
        else:
            self.df=self.df.iloc[-val_pct:,:]
            
            self.df_copy=self.df.copy()
            
    def __len__(self):
        return np.shape(self.df)[0]

    def __getitem__(self, idx):
        x=np.array(self.df.iloc[idx])  # randomised df
        
        y=np.array(self.df_copy.iloc[idx]) # target stays as orginal col
        
        sample = {'x': x, 'y': y}
        return sample

    

nn_layers=[400,320,250,200,250,320,400] # no input/out shapes

model=torch.nn.Sequential(torch.nn.Linear(np.shape(train_order)[1], nn_layers[0]) ,
                          
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[0]) ,                    
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(nn_layers[0], nn_layers[1]),  # 6*6 from image dimension
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[1])   ,
    torch.nn.Dropout(p=0.5),
                          
    torch.nn.Linear(nn_layers[1], nn_layers[2]),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[2])   ,
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(nn_layers[2], nn_layers[3]),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[3])   ,
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(nn_layers[3], nn_layers[4]) ,
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[4])   ,
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(nn_layers[4], nn_layers[5]) , # 6*6 from image dimension
                          
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[5])   ,
    torch.nn.Dropout(p=0.5),
                          
    torch.nn.Linear(nn_layers[5], nn_layers[6]),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm1d(num_features=nn_layers[6])   ,
    torch.nn.Dropout(p=0.5),
                          
    torch.nn.Linear(nn_layers[6], np.shape(train_order)[1])).cuda()


# he uses rank guass and only MSE
nn_layers=[1500,1500,1500] # no input/out shapes


model=torch.nn.Sequential(
    torch.nn.Linear(np.shape(train_order)[1], nn_layers[0]) ,              
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5),
    
    torch.nn.Linear(nn_layers[0], nn_layers[1]),  # 6*6 from image dimension
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5),
                          
    torch.nn.Linear(nn_layers[1], nn_layers[2]),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5),
    
    torch.nn.Linear(nn_layers[2], np.shape(train_order)[1])).cuda()
                          


model

# model.load_state_dict(torch.load("/kaggle/working/net400"))
# # torch.load()
# os.getcwd(),print(os.listdir("/kaggle/working/net400"))




num_binary=np.shape(binary_cols)[0] # number of 0-1 categories for BCELosses  

from torch.optim.lr_scheduler import StepLR

loss_fn = torch.nn.MSELoss()
binary_loss = torch.nn.BCELoss()
sigmoid=torch.nn.Sigmoid()
opt = torch.optim.Adam(model.parameters(), lr=init_lr)
scheduler = StepLR(opt, step_size=scheduler_step_size, gamma=0.99)


binary_loss(sigmoid(torch.tensor([-12.9])),sigmoid(torch.tensor([-12.9])))

def r(x): return(round(x,4))
r(3.2222222)


def trainer(model,epochs,bs=bs,swap=1):
    #bs is batch size, 
    #swap is number of epochs to recreate df with random swapping 
    L=[]
    V=[]
    c=[]
    b=[]
    
    counter_t=0
    counter_v=0
    
    for epoch in range(epochs):
        scheduler.step()
        
        if epoch%swap==0: 
            train_ds=NDataset_swap(swap_col_no)
            val_df=NDataset_swap(0,'val')

            data_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=2)  # restart loaders for randomization
            val_loader=DataLoader(val_df,bs,shuffle=True, num_workers=2) 
            loaders={'train':data_loader,'val': val_loader}
        if (epoch+1)%save_every==0:
                        checkpoint = {'train_loss': L, 'valid_loss':V,'MSE_train':c,'BinaryTrain':b,'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }

                        torch.save(checkpoint, '/home/Drive/rahul/PortoSeguroSafeDriverPrediction/'+save_initial+str(epoch)+'V'+str(r(sum(V)/len(V))) )
                        
                        print('saved----------')
        
        for e in loaders:
            if e=='train':  model.train() ; grad=True #
            else: model.eval() ; grad=False

            for idx, batch_data in enumerate(loaders[e]):
                if e=='train':counter_t+=1
                else:counter_v+=1
                    
                batch_input = Variable(batch_data['x'].float()).cuda()
                target=Variable(batch_data['y'].float()).cuda()

                pred=model(batch_input)

                c_loss = loss_fn(pred[: , :-num_binary], target[: , :-num_binary]) # numeric, ordinal cols
                binary=binary_loss(sigmoid(pred[:,-num_binary:]),target[: , -num_binary:]) # binary and one hot cols
                
                loss=c_loss+binary
                
                if e=='train':
                    L.append(loss.item())
                    c.append(c_loss.item())
                    b.append(binary.item())
                    
                    if idx%10==0:print('Train: ',epoch,idx,r(sum(c)/len(c)),r(sum(b)/len(b)),r(sum(L)/len(L)))
                    
                    
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                    
                    #TFBoard
                    writer.add_scalar('Total Train Loss', loss.item(), counter_t)
                    writer.add_scalar('MSE Train Loss', c_loss.item(), counter_t)
                    writer.add_scalar('Binary Loss', binary.item(), counter_t)

                    writer.add_scalar('Learning Rate', scheduler.get_lr()[-1], counter_t)
                        
                        
                else:
                    if epoch%swap==swap-1:  # only do validation at the end of a swap cycle
                        V.append(loss.item())
                        print("Validation: ",epoch,idx,sum(V)/len(V))
                    writer.add_scalar('Total Train LossV', loss.item(), counter_v)
                    writer.add_scalar('MSE Train LossV', c_loss.item(), counter_v)
                    writer.add_scalar('Binary LossV', binary.item(), counter_v)
                    
                    

    return(model,L,V,c,b)

model,L,V,co,bi=trainer(model,epochs_run)

checkpoint = {'train_loss': L, 'valid_loss':V,'MSE_train':co,'BinaryTrain':bi,'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }

torch.save(checkpoint, '/home/Drive/rahul/PortoSeguroSafeDriverPrediction/'+save_initial+str(epochs_run)+str(r(sum(V)/len(V)) ))
print('Saved : ', '/home/Drive/rahul/PortoSeguroSafeDriverPrediction/'+save_initial)
