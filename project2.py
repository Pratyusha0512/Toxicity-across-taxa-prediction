#!/usr/bin/env python
# coding: utf-8

# # INTENT

# The given dataset has the intention to create a predictive model that relates the chemical structures and properties of compounds to their toxicity towards fish. It aims to find the relationship between the molecular descriptors and properties with respect to the damage caused to the fish in order to find out aa solution to minimize it with help of the analysis of the dataset.

# #Importing libraries required and opening the csv file and storing it in a dataframe created called df

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
df=pd.read_csv('qsar_fish_toxicity.csv')
df


# #Data cleaning process starts.
# #1.Missing value imputation

# #As provided in the word file, the rows with value 0 of the column named 'SM1_Dz(Z)' are replaced by null value for better results

# In[2]:


df['SM1_Dz(Z)']=df['SM1_Dz(Z)'].replace(0,float('NaN'))
print(df.isna().sum())


# #The percentage of null values of each column as well as their skewness is calculated. The missing values are then imputed with median values as the dataset contains discrete and continuous variables.

# In[3]:


df.isna().sum()/len(df)*100


# In[4]:


df.skew()


# In[5]:


for i in df.columns :
        df[i].fillna(df[i].median(),inplace=True)
print(df.isna().sum())


# #2. Outliers are detected by using z score and IQR method and selecting the best i.e z score

# In[6]:


outliers1=pd.DataFrame()
df1=pd.DataFrame()
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR
    outliers1 = (df[col] < LB) | (df[col] > UB)
    df1=df[~outliers1]
df1.info()


# In[7]:


import copy
outliers2=pd.DataFrame()
df2=pd.DataFrame()
z_scores = np.abs((df - df.mean()) / df.std())
outliers2= (z_scores > 3)|(z_scores < -3)
df2=df[~outliers2]
df2.info()
df=copy.copy(df2)


# #Imputing outlier values with knn imputation for better result

# In[8]:


from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=6)
imputed_df =pd.DataFrame(knn_imputer.fit_transform(df),columns=df.columns)
df=copy.copy(imputed_df)
df.info()


# #Before continuing with the dataset, #3. Data scaling is performed as a data normalization technique for cleaner and more uniform data

# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df= pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df=copy.copy(scaled_df)
df


# #Finding its descriptive values and visualizing the dataset for proper observation

# In[10]:


df.describe()


# #From the visualization techniques heatmap and pairplot we can observe the correlation and decide what features to select as the X subset for training and testing for optimal observations. As we can see that the features 'MLOGP' and 'SM1_Dz(Z)' are highly correlated to the target variable 'LC50 [-LOG(mol/L)]'

# In[11]:


correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True,fmt='.2f')
plt.title("Heatmap")
plt.show()


# In[12]:


sns.pairplot(df) 


# #Now a boxplot helped us observe how different are the values of the outliers from other values. And as we can observe NdssC is heavily skewed. NdsCH contains more outliers than other features

# In[13]:


column_names=df.columns
plt.figure(figsize=(10, 6))
df.boxplot(vert=True)
plt.title('Boxplot of All Features')
plt.xlabel('Values')
plt.ylabel('Features')
plt.show()


# In[14]:


sns.histplot(df,bins=20)


# #We imported splitting function and the evaluation parameters for the regression models. Then we performed the PCA feature selection algorith to observe every columns variance with respect to the target variable. As we can see it keeps on decreasing as we iterate through the columns

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


# In[16]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP','LC50 [-LOG(mol/L)]']]
y = df['LC50 [-LOG(mol/L)]'] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_components = 7
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
if 'LC50' in df.columns:
    pca_df['LC50'] = y
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)


# #We performed hyperparameter tuning on various regression models by selecting 'MLOGP','SM1_Dz(Z)','CIC0','GATS1i' features 

# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lr = LinearRegression()
param_grid = {}
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Mean_squared_error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[18]:


from sklearn.linear_model import Ridge
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
ridge = Ridge()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0,100.0]  
}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[19]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lasso = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[20]:


from sklearn.ensemble import RandomForestRegressor
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
rf = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],         
    'max_depth': [None, 10, 20, 30],       
    'min_samples_split': [2, 5, 10],        
    'min_samples_leaf': [1, 2, 4],          
    'max_features': ['auto', 'sqrt'],       
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[21]:


from sklearn.tree import DecisionTreeRegressor
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
dt = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 5, 10, 15],            
    'min_samples_split': [2, 5, 10],           
    'min_samples_leaf': [1, 2, 4],             
    'max_features': ['auto', 'sqrt'],          
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[22]:


from sklearn.ensemble import AdaBoostRegressor
X = df[['MLOGP','SM1_Dz(Z)','CIC0','GATS1i']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator = DecisionTreeRegressor(max_depth=4)
adaboost = AdaBoostRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5, 1.0]   
}
grid_search = GridSearchCV(adaboost, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_adaboost = grid_search.best_estimator_
y_pred = best_adaboost.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[23]:


from sklearn.ensemble import GradientBoostingRegressor
X = df[['MLOGP','SM1_Dz(Z)','GATS1i','CIC0']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5],       
    'max_depth': [3, 5, 7],                  
    'min_samples_split': [2, 5, 10],         
    'min_samples_leaf': [1, 2, 4],           
}
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_gb = grid_search.best_estimator_
y_pred = best_gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[24]:


from sklearn.ensemble import BaggingRegressor
X = df[['MLOGP','SM1_Dz(Z)','GATS1i','CIC0']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator = DecisionTreeRegressor()
bagging = BaggingRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'max_samples': [0.5, 0.7, 0.9],          
}
grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_bagging = grid_search.best_estimator_
y_pred = best_bagging.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[25]:


from sklearn.svm import SVR
X = df[['MLOGP','SM1_Dz(Z)','GATS1i','CIC0']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
svr = SVR()
param_grid = {
    'kernel': ['linear', 'rbf', 'poly','sigmoid'],       
    'C': [0.1, 1.0, 10.0],                    
    'gamma': ['scale', 'auto', 0.1, 0.01],     
    'degree': [2, 3, 4]                        
}
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[26]:


from sklearn.neighbors import KNeighborsRegressor
X = df[['MLOGP','SM1_Dz(Z)','GATS1i','CIC0']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],        
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]                          
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[27]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
X = df[['MLOGP','SM1_Dz(Z)','GATS1i','CIC0']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
gpr.fit(X_train, y_train)
y_pred, sigma = gpr.predict(X_test, return_std=True)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# #We got the highest R2_score for kneighbors regression model.It suggests that knn model fits the data the best compared to other models and that the KNN model captures a substantial portion of the variance in the target variable and its predictions align well with the actual target values.

# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lr = LinearRegression()
param_grid = {}
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Mean_squared_error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[29]:


from sklearn.linear_model import Ridge
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
ridge = Ridge()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.00]  
}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[30]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lasso = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[31]:


from sklearn.ensemble import RandomForestRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
rf = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],         
    'max_depth': [30,40,50],       
    'min_samples_split': [2, 5, 10],        
    'min_samples_leaf': [1, 2, 4],          
    'max_features': ['auto', 'sqrt'],       
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[32]:


from sklearn.tree import DecisionTreeRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
dt = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 5, 10, 15],            
    'min_samples_split': [2, 5, 10],           
    'min_samples_leaf': [1, 2, 4],             
    'max_features': ['auto', 'sqrt'],          
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[33]:


from sklearn.ensemble import AdaBoostRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator = DecisionTreeRegressor(max_depth=4)
adaboost = AdaBoostRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5, 1.0]   
}
grid_search = GridSearchCV(adaboost, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_adaboost = grid_search.best_estimator_
y_pred = best_adaboost.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[34]:


from sklearn.ensemble import GradientBoostingRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5],       
    'max_depth': [3, 5, 7],                  
    'min_samples_split': [2, 5, 10],         
    'min_samples_leaf': [1, 2, 4],           
}
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_gb = grid_search.best_estimator_
y_pred = best_gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[35]:


from sklearn.svm import SVR
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
svr = SVR()
param_grid = {
    'kernel': ['linear', 'rbf', 'poly','sigmoid'],       
    'C': [0.1, 1.0, 10.0],                    
    'gamma': ['scale', 'auto', 0.1, 0.01],     
    'degree': [2, 3, 4]                        
}
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[36]:


from sklearn.neighbors import KNeighborsRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],        
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]                          
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[37]:


from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import filterwarnings
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
filterwarnings("ignore")
gpr.fit(X_train, y_train)
y_pred = gpr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[38]:


from sklearn.ensemble import BaggingRegressor
X = df[['MLOGP','SM1_Dz(Z)']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator = DecisionTreeRegressor()
bagging = BaggingRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'max_samples': [0.5, 0.7, 0.9],          
}
grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_bagging = grid_search.best_estimator_
y_pred = best_bagging.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# #After considering a subset of 'MLOGP','SM1_Dz(Z)' as the features, we got the highest r2_score as 59 for randomforest regression model. It suggests that for the given set of features this model fits the best compared to the others. This also suggests that the features have a strong relationship with the target variable and give good predictive performance and analysis wwhen its done using random forest regression model. 

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lr = LinearRegression()
param_grid = {}
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Mean_squared_error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[40]:


from sklearn.linear_model import Ridge
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
ridge = Ridge()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.00]  
}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[41]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
lasso = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r1 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[42]:


from sklearn.ensemble import RandomForestRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
rf = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],         
    'max_depth': [None, 10, 20, 30],       
    'min_samples_split': [2, 5, 10],        
    'min_samples_leaf': [1, 2, 4],          
    'max_features': ['auto', 'sqrt'],       
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[43]:


from sklearn.tree import DecisionTreeRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
dt = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 5, 10, 15],            
    'min_samples_split': [2, 5, 10],           
    'min_samples_leaf': [1, 2, 4],             
    'max_features': ['auto', 'sqrt'],          
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[44]:


from sklearn.ensemble import AdaBoostRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator = DecisionTreeRegressor(max_depth=4)
adaboost = AdaBoostRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5, 1.0]   
}
grid_search = GridSearchCV(adaboost, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_adaboost = grid_search.best_estimator_
y_pred = best_adaboost.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[45]:


from sklearn.ensemble import GradientBoostingRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],          
    'learning_rate': [0.01, 0.1, 0.5],       
    'max_depth': [3, 5, 7],                  
    'min_samples_split': [2, 5, 10],         
    'min_samples_leaf': [1, 2, 4],           
}
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_gb = grid_search.best_estimator_
y_pred = best_gb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[46]:


from sklearn.svm import SVR
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
svr = SVR()
param_grid = {
    'kernel': ['linear', 'rbf', 'poly','sigmoid'],       
    'C': [0.1, 1.0, 10.0],                    
    'gamma': ['scale', 'auto', 0.1, 0.01],     
    'degree': [2, 3, 4]                        
}
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[47]:


from sklearn.neighbors import KNeighborsRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],        
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]                          
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[48]:


from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import filterwarnings
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
filterwarnings("ignore")
gpr.fit(X_train, y_train)
y_pred = gpr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# In[49]:


from sklearn.ensemble import BaggingRegressor
X = df[['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']]
y = df['LC50 [-LOG(mol/L)]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
estimator= DecisionTreeRegressor()
bagging = BaggingRegressor(estimator=estimator)
param_grid = {
    'n_estimators': [50, 100, 200],          
    'max_samples': [0.5, 0.7, 0.9],          
}
grid_search= GridSearchCV(bagging, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_bagging = grid_search.best_estimator_
y_pred = best_bagging.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse) ** 0.5
print("Root Mean Squared Error:", rmse)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
r2 = r2_score(y_test, y_pred)
print("R2_score:",r2 )


# #From the above set of subset of features 'CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP' where we have considered all the features excluding the target variable, we can observe that Bagging regression model fits the best with the data giving a r2_score of approximately 67%.
# 

# #From all our observations above it is hence concluded that the model with the highest R2_score of '66%' is Bagging regression model which takes into consideration a subset of all the features excluding the target variable.
# A low R2_score might mean that the model is underfitted and similarly high R2_score might mean that the model is overfitted.
# But a point to be noted is that a low or a high r2_score does not necessarily imply whether the model is a perfect fit for the given dataset ,whether it gives a good predictive performance or whether the features have a strong relationship with the target variable.
# The goodness of a R2_score usually depends on the complexity of the dataset.If the dataset is very complex then even a R2_score of 50% for a model would be good enough as a fit for further observations and solving the problem with a proper conclusion.

# In[ ]:




