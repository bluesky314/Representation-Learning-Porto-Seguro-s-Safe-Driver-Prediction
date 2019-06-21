# Representation Learning using Embedding Layers in PyTorch for Porto Seguro’s Safe Driver Prediction Challenge

Creating an embedding network for automatic feature engineering in the Porto Seguro’s Safe Driver Prediction Challenge.
All categorial and ordinal features are embedded and processed along with numeric features for final output.

All embeddings layers and dense layers are made in a loop. The network is extremely flexible to different number of layers and embeddings. This serves as a very reusable code piece for future competitions. Only basic feature engineering such as outlier detection, missing values, normalization and one-hot encoding is required

Network Architecture:

<img src="https://camo.githubusercontent.com/f8ef85636f11960c7b85d465a9844695480ff37f/68747470733a2f2f6769746875622e636f6d2f7869616f7a686f7577616e672f6b6167676c652d706f72746f2d73656775726f2f7261772f383364373934663664636536333234366165663637323039626635393662646165353466656132322f4a7570797465725f6e6e6d6f64656c2f4a7570797465725f696d6167652f4e4e5f6c617965722e706e67" width="700" height="200">


1)

We put all category column names in a list.

Then we create an embedding layer for each of these depending upon that feature's unique value count in a loop and store them in the dictionary self.cat_dict

Each embedding layer is saved in the dictionary according to its feature name so it can be retrieved later

2)

We process the remaining numeric features through any number or size of fc layers given by fc_layers argument

This is done in part so thhe number of embeddings dont vastly outnumber numeric features

3)

We then concatonate the embeddings and fc layers and pass through another set of fc layers of any number or size given by merge_layers argument


-----
# Denoising Autoencoder(DAE)

We also create a Denoising Autoencoder(DAE) to learn unsupervised representation of our data that can later be used in a supervised model. This model was used to win the Porto Seguro’s Safe Driver Prediction Challenge with some interesting tricks including "swap noise" data augmentation and GaussRank Normalization: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629 . I learnt alot from this thread and it shows that neural networks are very powerful for tabular data and is a good model to have in ones arsenal. One just needs to know how to train them. The advantage here is once again automated feature generation when features are hard to create which they were as this data was annomozed. 

We have 38 features initially and after one-hot encoding it becomes 185 

Swap Noise: In order to create artifical noise or data augmentation for tabular data we use 'swap noise' Here we sample and swap from the feature itself with a certain probability. So a probability 0.15 means 15% of features in a row are replaced by values from another row. (https://arxiv.org/pdf/1801.07316.pdf)
 
GaussRank Normalization: As after one-hot encoding we have around 8 numeric features and the rest are binary or ordinal. Neural networks learn must more efficently when data is normalised and ordinal variables cannot be normalised into gaussian by standard methods but GaussRank can forces it to be normal. What the big deal? With GaussRank I reached a loss half in 2 epochs of that with standard normalization in 500 epochs! I was amazed by the difference. After discussing with my senior I was told it was due to the large presence of categorical variables which made the optimization plane non-smooth. 
Read more : http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss/

GaussRank Proceadure:
First we compute ranks for each value in a given column (argsort). 
Then we normalize the ranks to range from -1 to 1. 
Then we apply the mysterious erfinv function. (https://wiki.analytica.com/index.php?title=ErfInv)

While the winner only used plain MSE loss for all variables, I also experiment with MSE + Binary CE for binary variables. 
