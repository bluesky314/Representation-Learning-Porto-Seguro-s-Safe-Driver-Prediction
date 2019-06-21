

bs=212
init_lr=0.001
scheduler_step_size = 10

save_every=50# save every _ epochs

epochs_run=1000 # No of epochs

save_initial='LoopedEmbedFirst'
import os
gpu_number = "1"            

from tensorboardX import SummaryWriter
writer = SummaryWriter()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

# print(os.listdir("../input/"))
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
            role = 'target'
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
meta.drop(['target'],inplace=True)
meta.drop(['keep'],axis=1,inplace=True)
meta.drop(['role'],axis=1,inplace=True)




# Feature Engineering

df = pd.read_csv('/home/Drive/rahul/PortoSeguroSafeDriverPrediction/train.csv')

df.drop(['id'],axis=1,inplace=True)
orig=df.copy()

df=df.iloc[:50000,:]

df.target.value_counts()

# removing calc cols increases baseline xgboost from 0.22 to 0.23227886570
for e in df.columns:
    if 'calc' in e: df.drop(e,axis=1,inplace=True)

meta=create_meta(df)
meta.drop(['target'],inplace=True)
meta.drop(['keep'],axis=1,inplace=True)
meta.drop(['role'],axis=1,inplace=True)

np.shape(df)

# Handle missing vals first

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

# # Make new feat as heavy concentration after 102
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

# # df.drop(['ps_car_11_cat'],axis=1,inplace=True) # drop this as it has wayyy to many categories 104

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


# normalise continous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train[meta[meta.level=='interval'].index] = scaler.fit_transform(train[meta[meta.level=='interval'].index])


np.shape(np.concatenate([ meta[meta.level=='binary'].index,meta[meta.level=='nominal'].index ])),np.shape(meta[meta.level=='interval'].index),np.shape(meta[meta.level=='ordinal'].index)

# ![](http://)

# Embedding Network 

# As there is heavy class imbalance, only 3.6% is class 1, we do upsampling of minority class




print('Before',np.shape(train))

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
print('After')
print(df_upsampled.target.value_counts())

train=df_upsampled
y=train.target
# train=train.drop(['target'],axis=1)
print(np.shape(train))



# since features are anonymised into 3 categories, we split them into category-type

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

type(df.ps_reg_01[0])

# Split each category into numeric or category

ind_cols_cat=[]                      #### .   FOR THIS MODEL, WE INCLUDE ORDINAL IN CATEGORY COLS TO ENCODE THEM
ind_cols_num=[]
for f in ind_cols:
    if 'bin' in f or f == 'target':
        ind_cols_cat.append(f)
    elif 'cat' in f or f == 'id':
        ind_cols_cat.append(f)
    elif df[f].dtype == float:
        ind_cols_num.append(f)
    elif df[f].dtype == int: ## INT are ordinal
        ind_cols_cat.append(f)
        
        
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
        reg_cols_cat.append(f)
        
        
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
        car_cols_cat.append(f)

        

all=[np.array(ind_cols_cat),np.array(ind_cols_num),np.array(reg_cols_cat),np.array(reg_cols_num),np.array(car_cols_cat),
     np.array(car_cols_num)]
for e in all:
    print(np.shape(e))

cat_cols=ind_cols_cat+reg_cols_cat+car_cols_cat
nums_cols=ind_cols_num+reg_cols_num+car_cols_num

len(cat_cols),len(nums_cols)

g=[]
for e in cat_cols:
  g.append(np.shape(np.unique(train[e]))[0])
#   print(e,np.shape(np.unique(train[e]))[0])
np.unique(g)

from torch.utils.data import Dataset, DataLoader

class LoopDataset(Dataset):
    """Returns input_vals_cat as dict with column keys and numerics as tesnor"""

    def __init__(self, df,y,type="train",pct=0.85): # 
       
        N=np.shape(df)[0]
        train_pct=int(pct*N)
        val_pct=int((1-pct)*N)
        
        
        if type=="train":
            self.df=df.iloc[:train_pct,:]
            self.y=y[:train_pct]
        else:
            self.df=df.iloc[-val_pct:,:]
            self.y=y[-val_pct:]
            
            
    def __len__(self):
        return np.shape(self.df)[0]

    def __getitem__(self, idx,cat_cols=cat_cols,nums_cols=nums_cols):
      
        input_vals_cat={}
        for e in cat_cols:
          input_vals_cat[e]= torch.tensor(self.df[e].iloc[idx]).long().unsqueeze(0)

        input_num=[]
        for e in nums_cols:
          input_num.append(self.df[e].iloc[idx]) 
        input_num=torch.tensor(input_num) 
        
        target=torch.tensor(self.y.iloc[idx])
        
        sample = {"input_vals_cat":input_vals_cat,"input_num":input_num,"target":target }
        return sample



import torch.nn.functional as F

class loop_embednet(torch.nn.Module):

    def __init__(self,df,fc_layers=[100,200,300],merge_layers=[512,64,1],small_embed=5,big_embed=9,cat_list=cat_cols,num_list=nums_cols):
      
        # df is train dataset
        # fc_layers is sizes of fc layers for numeric excluding input size
        # merge_layers is sizes of fc layers for concatonated embedding and fc_layer output excluding input size
        # cat_list is list of category cols to create embedding layers for
        # num_list is list of numeric cols to use for fc layers
        
        
        super(loop_embednet, self).__init__()

        
        self.cat_list=cat_cols
        self.num_list=num_list
        merge_count=fc_layers[-1] # to count the inputs into first merge layer as total embedings + last fc
        
  
        self.cat_dict={} # dict of embedding layers in a loop with key as feature_layer
        
        
        
        for e in cat_list:
          unique=np.shape(np.unique(df[e]))[0] 
          embedding_dim = small_embed if np.shape(np.unique(df[e]))[0]==2 else big_embed # for binary use small_embed embeddings, else big_embed
          self.cat_dict[e+'_layer']=torch.nn.Embedding(unique, embedding_dim) # save layer as feature name
          self.cat_dict[e+'_drop']=torch.nn.Dropout(p=0.5)
          merge_count+=embedding_dim
          
          
                    
          
        self.fc_layers=fc_layers                  
        self.fc_layers.insert(0, len(num_list) ) # add input size to first layer
        
        for i in range(len(self.fc_layers)-1):
            setattr(self, 'fc'+str(i), torch.nn.Linear(self.fc_layers[i], self.fc_layers[i + 1]))
            setattr(self, 'drop'+str(i), torch.nn.Dropout(p=0.5))
            
        
               
        
        self.merge_layers=merge_layers                  
        self.merge_layers.insert(0, merge_count ) # add input size as merge_count
        
        for i in range(len(self.merge_layers)-1):
          setattr(self, 'merge'+str(i), torch.nn.Linear(self.merge_layers[i], self.merge_layers[i + 1]))
          setattr(self, 'dropm'+str(i), torch.nn.Dropout(p=0.5))
          


    def forward(self, input_vals_cat,numeric_vals): 
      # input_vals_cat is dict with keys of category, and values as batches
      # numeric is batch * concatonated vals of all numerics
        
        var = [] 
        for e in self.cat_list:
          layer=self.cat_dict[e+'_layer']
          dropout=self.cat_dict[e+'_drop']

          layer_output=dropout(layer(input_vals_cat[e])) # Activation - tanh/relu
          var.append( layer_output.view(layer_output.size(0), -1) )
        
                  

        var_tensor = torch.cat(var,dim=1)
    
        for i in range(len(self.fc_layers)-1):
          layer = getattr(self, 'fc'+str(i))
          dropout=getattr(self, 'drop'+str(i))
          numeric_vals = dropout(F.relu(layer (numeric_vals)))

          
        
        merge=torch.cat((var_tensor,numeric_vals),dim=1)

        
        for i in range(len(self.merge_layers)-1):
          layer = getattr(self, 'merge'+str(i) )
          dropout=getattr(self, 'dropm'+str(i))
          if i!=len(self.merge_layers)-2:merge = dropout(F.relu(layer (merge))) # if to prevent last output from relu
          else: merge = layer (merge)
            
          

          

        
        
        return(merge)

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
device

model=loop_embednet(train,fc_layers=[32,128],merge_layers=[512,64,1],small_embed=6,big_embed=6).to(device)

for e in model.cat_dict.values():  # IMPORTANT, model.cuda() will not set dictionary list to cuda
  e=e.to(device)




ds=LoopDataset(train,y)
data_loader_train = DataLoader(ds, bs, shuffle=True, num_workers=2)

ds_val=LoopDataset(train,y,type='val')
data_loader_val = DataLoader(ds_val, bs, shuffle=True, num_workers=2)


loaders={'train':data_loader_train,'val': data_loader_val}


from torch.optim.lr_scheduler import StepLR



init_lr=0.0003
binary_loss = torch.nn.BCELoss()

sigmoid=torch.nn.Sigmoid()

opt = torch.optim.Adam(model.parameters(), lr=init_lr,weight_decay=1e-3)

scheduler = StepLR(opt, step_size=scheduler_step_size, gamma=0.90)


def r(x): return(round(x,4))
r(3.2222222)




def trainer_embed(model,epochs,L=0,V=0):
    if L==0 and V==0:
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
              
                for j in batch_data["input_vals_cat"].keys():
                  batch_data["input_vals_cat"][j]=batch_data["input_vals_cat"][j].to(device)
                
                target=Variable(batch_data['target']).float().to(device)
                pred=model(batch_data['input_vals_cat'],batch_data['input_num'].to(device)) 
                
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
                    
                    writer.add_scalar('Total Train Loss', loss.item(), counter_t)
                    writer.add_scalar('Learning Rate', scheduler.get_lr()[-1], counter_t)
#                     writer.add_scalar('MSE Train Loss', c_loss.item(), counter_t)
                    
#                     print('p',pred[:10],np.shape(pred))
#                     print('t',target.detach().cpu().numpy()[:10],np.shape(target))
                    print(e,'Epoch:',epoch,'Batch:',idx,'Loss:',sum(L)/len(L))#,'Gini:',eval_gini(target.detach().cpu().numpy(), pred))
                    
                    
                else:
                    V.append(loss.item())
                    
                    pred=pred.squeeze(1).detach().cpu().numpy()
                    pred[pred<0.5]=0
                    pred[pred>0.5]=1
                    writer.add_scalar('Total Train LossV', loss.item(), counter_v)
                    print(e,'Epoch:',epoch,'Batch:',idx,'Loss:',sum(V)/len(V),'Gini:',eval_gini(target.detach().cpu().numpy(), pred))
    return(model,L,V)

model,L,V=trainer_embed(model,epochs_run)