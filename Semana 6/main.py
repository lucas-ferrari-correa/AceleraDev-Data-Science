#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[3]:


# Algumas configurações para o matplotlib.

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


fifa = pd.read_csv("fifa.csv")


# In[10]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[7]:


fifa.head()


# In[8]:


fifa.describe()


# In[11]:


fifa.shape


# In[9]:


fifa.info()


# In[12]:


fifa.isnull().sum()


# In[13]:


fifa.dropna(inplace=True)


# In[14]:


pca = PCA(n_components = 3)

plot = pca.fit_transform(fifa)
print(plot)


# In[15]:


sns.scatterplot(plot[:, 0], plot[:, 1], plot[:, 2])


# In[18]:


pca = PCA(n_components = 2)

plot = pca.fit_transform(fifa)
print(plot)


# In[16]:


sns.scatterplot(plot[:, 0], plot[:, 1])


# In[23]:


pca = PCA().fit(fifa)
evr = pca.explained_variance_ratio_
print("evr = " + str(evr))
print("len = " + str(len(evr)))


# In[29]:


x_evr = np.arange(len(evr))
y_evr = np.cumsum(evr)

print("x_evr = " + str(x_evr))
print("y_evr = " + str(y_evr))


# In[32]:


graph_evr = sns.lineplot(x_evr, y_evr)
graph_evr.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[17]:


pca = PCA().fit(fifa)

evr = pca.explained_variance_ratio_
x_evr = np.arange(len(evr))
y_evr = np.cumsum(evr)

graph_evr = sns.lineplot(x_evr, y_evr)
graph_evr.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# In[39]:


print(evr)


# In[40]:


def q1():
    first_component = round(evr[0], 3)
    
    return float(first_component)
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[41]:


def q2():
    evr_cum = evr.cumsum()
    
    i = 0
    for x in range(0, len(evr) + 1):
        i += 1
        if(evr_cum[x] > 0.95):
            break
    return int(i)
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[42]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[43]:


len(x)


# In[44]:


pca = PCA(2).fit(fifa)
print(pca)


# In[45]:


pca.components_.dot(x).round(3)


# In[46]:


def q3():
    pca = PCA(2).fit(fifa)
    result = pca.components_.dot(x).round(3)
    
    return tuple(result)
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[5]:


fifa['Overall']


# In[18]:


X = fifa.drop('Overall', 1)
y = fifa['Overall']


# In[19]:


def q4():
    lin_reg = LinearRegression()
    rfe = RFE(lin_reg, n_features_to_select = 5)
    rfe.fit(X, y)
    
    mask = rfe.support_
    result = X.columns[mask]
    
    return list(result)
q4()

