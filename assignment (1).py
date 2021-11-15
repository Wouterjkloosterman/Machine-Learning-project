#!/usr/bin/env python
# coding: utf-8

# ## Machine learning assignment

# In[1]:


# sources
# https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/


# In[2]:


#importing the necessary data and packages

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dftrain = pd.read_json("train-1.json")
dftest = pd.read_json("test.json")


# In[ ]:





# In[ ]:





# In[3]:


#create the dataframe
dftrain = pd.DataFrame(dftrain)
dftest = pd.DataFrame(dftest)


# In[ ]:





# In[4]:


#display the first 4 cases
display(dftrain.head())
display(dftest.head())


# In[ ]:





# In[5]:


#print the shapes, how many variables and observations
print(dftrain.shape)
print(dftest.shape)


# In[6]:


#get some info about the dataset
dftrain.info


# In[7]:


dftrain.describe


# In[8]:


#detect missing values
dftrain.isnull().sum()


# In[9]:


#change format authors
dftrain["authors"] = [",".join(i) for i in dftrain["authors"]]
dftest["authors"] = [",".join(i) for i in dftest["authors"]]


# In[10]:


#change format topics
dftrain["topics"] = [",".join(i) for i in dftrain["topics"]]
dftest["topics"] = [",".join(i) for i in dftest["topics"]]


# In[11]:


dftrain.head()


# In[12]:


#Checking for the most frequent year in the dataset
freq = dftrain["year"].value_counts()
print(freq)


# In[13]:


plt.figure(figsize=(10,5))
sns.countplot("year", data=dftrain)
plt.xticks(rotation=90)
plt.show()


# In[14]:


#fix the three missing years
dftrain["year"] = dftrain["year"].fillna(2019.0).astype(int)
dftest["year"] = dftest["year"].fillna(2019.0).astype(int)


# In[ ]:





# In[15]:


#Check if method works
dftrain.isnull().sum()


# In[16]:


# fill missing abstracts with no text
dftrain["abstract"] = dftrain["abstract"].fillna("")
dftest["abstract"] = dftest["abstract"].fillna("")


# In[17]:


#Check if method works
dftrain.isnull().sum()


# In[18]:


#value that occurs most in field of study
mode1 = dftrain["fields_of_study"].mode().values[0]


# In[19]:


print(mode1)


# In[20]:


#fill missing values fields_of_study
dftrain["fields_of_study"] = dftrain["fields_of_study"].apply(lambda d: d if isinstance(d, list) else mode1)
dftest["fields_of_study"] = dftest["fields_of_study"].apply(lambda d: d if isinstance(d, list) else mode1)


# In[21]:


dftrain["fields_of_study"] = ["".join(i) for i in dftrain["fields_of_study"]]
dftest["fields_of_study"] = ["".join(i) for i in dftest["fields_of_study"]]


# In[22]:


#Check if method works
dftrain.isnull().sum()


# In[23]:


#visualize the duplicates within dataset
duplicate = dftrain.duplicated()
print(duplicate.sum())
duplicate = dftest.duplicated()
print(duplicate.sum())


# In[24]:


#drop the duplicates
dftrain.drop_duplicates(inplace=True)
dftest.drop_duplicates(inplace=True)


# In[25]:


#check if duplicates are dropped
dpl = dftrain.duplicated()
print(dpl.sum())
dpl = dftest.duplicated()
print(dpl.sum())


# In[26]:


#Removing the columnns we don't need

venues = dftrain["venue"]
print(dftrain.dtypes)
#see how much unique outcomes
print (dftrain['venue'].unique()) #with this code you see a lot of unique 
#in my opinion to much and worthless, so drop them
to_drop = ["venue"]
dftrain.drop(to_drop, inplace=True, axis=1)
dftest.drop(to_drop, inplace=True, axis=1)


# In[27]:


#make everything lower letters

#for train
dftrain["title"] = dftrain["title"].str.lower()
dftrain["abstract"] = dftrain["abstract"].str.lower()
dftrain["authors"] = dftrain["authors"].str.lower()
dftrain["topics"] = dftrain["topics"].str.lower()
dftrain["fields_of_study"] = dftrain["fields_of_study"].str.lower()
#for test
dftest["title"] = dftest["title"].str.lower()
dftest["abstract"] = dftest["abstract"].str.lower()
dftest["authors"] = dftest["authors"].str.lower()
dftest["topics"] = dftest["topics"].str.lower()
dftest["fields_of_study"] = dftest["fields_of_study"].str.lower()


# In[28]:


#make dataframes in same order for comparison reasons
dftest = dftest[["doi", "title", "abstract", "authors", "year", "references", "topics", "is_open_access", "fields_of_study"]]
display(dftrain.head())
display(dftest.head())


# In[ ]:





# In[29]:


#making all variables the right type
dftrain.dtypes
#wee se a view strange categories, but change the dtype to category
pd.value_counts(dftrain.topics)
pd.value_counts(dftrain.fields_of_study)

dftrain["topics"]=dftrain["topics"].astype("category")
dftrain["fields_of_study"]=dftrain["fields_of_study"].astype("category")

#check if it worked
dftrain.dtypes


# ## Cleaning the columns

# In[30]:


# Title


# In[31]:


pd.value_counts(dftrain.title)


# In[ ]:





# In[ ]:





# In[32]:


# Abstract


# In[33]:


# I see duplicates in column
pd.value_counts(dftrain.abstract)


# In[2]:


dftrain["abstract"].value_count


# In[34]:


print(dftrain.shape)
# Count duplicate on a column
dftrain.abstract.duplicated().sum()


# In[35]:


#drop the duplicates in a colum, not working at the moment
dftrain = dftrain.drop_duplicates(subset=["abstract"], keep = "last")
print(dftrain.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


# Authors


# In[ ]:





# In[ ]:





# In[37]:


# Topics
#see the different topics categories
pd.value_counts(dftrain.topics)
#How to deal with this, there are a lot topics only once in the list


# ## Dealing with outliers

# In[38]:


#we can do iqr method, z score....
#there are a lot of methods, how we want to deal important but need to decide
#make boxplots and historgrams for nummeric variables: year, references and citations
dftrain["year"].hist(bins=100)


# In[39]:


#make boxplot to study feature closer
dftrain.boxplot(column=["year"])


# In[ ]:





# In[40]:


#For categorical variables, we have the bar chart
#dftrain["topics"].value_counts().plot.bar()


# In[ ]:





# In[41]:


#removing outliers using IQR
#check outliers
#sns.boxplot(data=dftrain, x=dftrain["year"])


# In[42]:


#implementation
#Q1=dftrain["year"].quantile(0.25)
#Q3=dftrain["year"].quantile(0.75)
#IQR=Q3-Q1
#print(Q1)
#print(Q3)
#print(IQR)
#Lower_Whisker = Q1 - 1.5 * IQR
#Upper_Whisker = Q3 + 1.5 * IQR
#print(Lower_Whisker, Upper_Whisker)


# In[43]:


#Outlier treatment
#dftrain = dftrain[dftrain["year"]< Upper_Whisker]
#dftrain = dftrain[dftrain["year"]> Lower_Whisker]
#Outliers will be any points below Lower_Whisker or above Upper_Whisker


# In[44]:


#see outliers
#sns.boxplot(data=dftrain, x=dftrain["citations"])


# In[45]:


#another technique is the descriptive statistics
#dftrain["citations"].describe()


# In[46]:


#from scipy import stats
#Z score
#Get the z-score table
#z = np.abs(stats.zscore(dftrain.citations))
#print(z)


# In[47]:


#We find the z-score for each of the data point in the dataset and if the z-score is greater than 3 than we can classify that point as an outlier. Any point outside of 3 standard deviations would be an outlier.
#threshold = 2
#print(np.where(z>2))


# In[48]:


#dftrain = dftrain[(z<2)]


# In[49]:


#see plot after treatment
#sns.boxplot(data=dftrain, x=dftrain["citations"])


# 'year'

# In[50]:


#see outliers
#sns.boxplot(data=dftrain, x=dftrain["year"])


# In[51]:


#another technique is the descriptive statistics
#dftrain["year"].describe()


# In[52]:


#from scipy import stats
#Z score
#Get the z-score table
#z = np.abs(stats.zscore(dftrain.year))
#print(z)


# In[53]:


#We find the z-score for each of the data point in the dataset and if the z-score is greater than 3 than we can classify that point as an outlier. Any point outside of 3 standard deviations would be an outlier.
#threshold = 2
#print(np.where(z>2))


# In[54]:


#dftrain = dftrain[(z<2)]


# In[55]:


#see plot after treatment
#sns.boxplot(data=dftrain, x=dftrain["year"])


# "references"

# In[56]:


#see outliers
#sns.boxplot(data=dftrain, x=dftrain["references"])


# In[57]:


#another technique is the descriptive statistics
#dftrain["year"].describe()


# In[58]:


#from scipy import stats
#Z score
#Get the z-score table
#z = np.abs(stats.zscore(dftrain.references))
#print(z)


# In[59]:


#We find the z-score for each of the data point in the dataset and if the z-score is greater than 3 than we can classify that point as an outlier. Any point outside of 3 standard deviations would be an outlier.
#threshold = 2
#print(np.where(z>2))


# In[60]:


#dftrain = dftrain[(z<2)]


# In[61]:


#see plot after treatment
#sns.boxplot(data=dftrain, x=dftrain["references"])


# In[62]:


# Check how many observations are hold
#print(dftrain.shape)


# ## Method quantille
# #https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame

# In[63]:


import matplotlib.pyplot as plt
plt.boxplot(dftrain["references"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample references')
print(dftrain.shape)

q = dftrain["references"].quantile(0.95)

dftrain[dftrain["references"] < q]

q_low = dftrain["references"].quantile(0.05)
q_hi  = dftrain["references"].quantile(0.95)

print(q_low)
print(q_hi)

dftrain = dftrain[(dftrain["references"] < q_hi) & (dftrain["references"] > q_low)]

print(dftrain.shape)


# In[64]:


plt.boxplot(dftrain["citations"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample citations')
print(dftrain.shape)

q = dftrain["citations"].quantile(0.95)

dftrain[dftrain["citations"] < q]

q_low = dftrain["citations"].quantile(0.05)
q_hi  = dftrain["citations"].quantile(0.95)

print(q_low)
print(q_hi)

dftrain = dftrain[(dftrain["citations"] < q_hi) & (dftrain["citations"] > q_low)]

print(dftrain.shape)


# In[65]:


plt.boxplot(dftrain["year"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample year')
print(dftrain.shape)

q = dftrain["year"].quantile(0.95)

dftrain[dftrain["year"] < q]

q_low = dftrain["year"].quantile(0.05)
q_hi  = dftrain["year"].quantile(0.95)

print(q_low)
print(q_hi)

dftrain = dftrain[(dftrain["year"] < q_hi) & (dftrain["year"] > q_low)]

print(dftrain.shape)


# In[66]:


plt.boxplot(dftrain["references"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample references')


# In[67]:


plt.boxplot(dftrain["citations"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample citations')


# In[68]:


plt.boxplot(dftrain["year"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample year')


# In[69]:


print(dftrain.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Boxplot method

# In[ ]:





# In[70]:


#references = dftrain_filtered["references"]

#import matplotlib.pyplot as plt
#plt.boxplot(references, vert=False)
#plt.title("Detecting outliers using Boxplot")
#plt.xlabel('Sample')


# ## Visualizing data

# In[71]:


#checking for correlation
dftrain.corr()


# In[ ]:





# In[ ]:





# In[72]:


sns.heatmap(dftrain.corr(), annot=True)


# In[73]:


dftrain[["year", "citations", "references", "is_open_access"]].mean()


# In[74]:


dftrain[["year", "citations", "references", "is_open_access"]].median()


# In[75]:


dftrain[["year", "citations", "references"]].describe()


# In[76]:


dftrain["is_open_access"].describe()


# In[77]:


dftrain.agg(
    {
        "year": ["min", "max", "median", "mean"],
        "citations": ["min", "max", "median", "mean"],
        "references": ["min", "max", "median", "mean"],
    }
)


# In[78]:


dftrain.drop(['title', 'abstract', 'is_open_access', 'authors', 'topics', 'fields_of_study'], axis=1).plot.line(title='Dataset')
#doesnt say something, so on itself representation


# In[79]:


# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(dftrain['year'])
# set title and labels
ax.set_title('Year frequencies in dataset')
ax.set_xlabel('Year')
ax.set_ylabel('Frequency')


# In[80]:


# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(dftrain['citations'])
# set title and labels
ax.set_title('Citation frequencies in dataset')
ax.set_xlabel('Citations')
ax.set_ylabel('Frequency')


# In[81]:


# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(dftrain['references'])
# set title and labels
ax.set_title('Reference frequencies in dataset')
ax.set_xlabel('References')
ax.set_ylabel('Frequency')


# In[ ]:





# In[82]:


dftrain.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)


# In[83]:


#dftrain.groupby("topics").citations.mean().sort_values(ascending=False).plot.bar()


# In[84]:


dftrain["title"].value_counts()


# In[85]:


#calculate the percentage of each education category.
#dftrain.is_open_access.value_counts(normalize=True)

#plot the pie chart of education categories
#dftrain.is_open_access.value_counts(normalize=True).plot.pie()
#plt.show()


# In[86]:


dftrain["abstract"].value_counts()


# In[87]:


dftrain["authors"].value_counts()


# In[88]:


dftrain["topics"].value_counts()


# In[ ]:





# In[89]:


dftrain["fields_of_study"].value_counts()


# In[90]:


dftrain.groupby('is_open_access')['year'].mean()


# In[91]:


dftrain.groupby('is_open_access')['year'].median()


# In[92]:


sns.boxplot(dftrain.is_open_access, dftrain.year)
plt.show()


# In[93]:


dftrain.groupby('is_open_access')['references'].mean()


# In[94]:


dftrain.groupby('is_open_access')['references'].median()


# In[95]:


sns.boxplot(dftrain.is_open_access, dftrain.references)
plt.show()


# In[96]:


dftrain.groupby('is_open_access')['citations'].mean()


# In[97]:


dftrain.groupby('is_open_access')['citations'].median()


# In[98]:


sns.boxplot(dftrain.is_open_access, dftrain.citations)
plt.show()


# In[ ]:





# In[ ]:





# ## TFIDF

# In[99]:


#TFIDF nice example https://www.analyticsvidhya.com/blog/2021/09/creating-a-movie-reviews-classifier-using-tf-idf-in-python/


# In[100]:


text = dftrain["abstract"]


# In[101]:


import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[102]:


#using the count vectorizer
count = CountVectorizer()
word_count=count.fit_transform(text)
print(word_count)


# In[103]:


word_count.shape # so 6232 abstracts and 19580 unique words


# In[104]:


#print(word_count.toarray())


# In[105]:


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count.get_feature_names(),columns=["idf_weights"])


# In[106]:


#inverse document frequency
df_idf.sort_values(by=['idf_weights'])


# In[107]:


#tfidf
tf_idf_vector=tfidf_transformer.transform(word_count)
feature_names = count.get_feature_names()


# In[108]:


first_document_vector=tf_idf_vector[1]
df_tfifd= pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])


# In[109]:


df_tfifd.sort_values(by=["tfidf"],ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


def words(text):
    l = []
    for i in text.split():
        if i not in l:
            k = l.append(i)
    return l


# In[111]:


import requests
import operator


def start(url):
    word_list = []
    source_code = abstracts
    soup = BeautifulSoup(source_code)
    for post_text in soup.findAll('a', {'class': 'title'}):
        content = post_text.string
        words = content.lower().split()
        for each_word in words:
            word_list.append(each_word)
    clean_up_list(word_list)


def clean_up_list(word_list):
    clean_word_list = []
    for word in word_list:
        symbols = "!@#$%^&*()_+{}:\"<>?,./;'[]-='"
        for i in range(0, len(symbols)):
            word = word.replace(symbols[i], "")
        if len(word) > 0:
            clean_word_list.append(word)
    create_dictionary(clean_word_list)


def create_dictionary(clean_word_list):
    word_count = {}
    for word in clean_word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    for key, value in sorted(word_count.items(), key=operator.itemgetter(1)):
        print(key, value)


# In[112]:


#print(create_dictionary(abstracts))


# In[113]:


#put variable abstract in a list
abstract = dftrain["abstract"]
abstrastlist = abstract.tolist()


# In[114]:


from collections import Counter
counts = Counter(abstrastlist)


# In[115]:


from collections import Counter
list1= abstrastlist
b=[]
for i in list1:
 b+=i.split(" ")
counts = Counter(b)
#print(counts)


# In[ ]:


from collections import Counter
from collections import *
ignore = ['hello', 'egg', 'the', 'of', 'and', 'a', 'to', 'in', 'for', 'that', 'on', 'is', 'we', 'this', 'with', 'our', 'as', 'are', 'by', 'from', 'an', 'which', 'can', 'be', 'show', 'it', 'than']
list1=abstrastlist
ArtofWarLIST=[]
for i in range(len(list1)):
    list1[i] = list1[i].lower()
for i in list1:
 ArtofWarLIST+=i.split(" ")
print(ArtofWarLIST)
ArtofWarCounter = Counter(x for x in ArtofWarLIST if x not in ignore)
#print(ArtofWarCounter)


# In[ ]:



#print(type(abstract))
#print(abstrastlist)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




