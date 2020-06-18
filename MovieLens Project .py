#!/usr/bin/env python
# coding: utf-8

# ### Project - MovieLens Data Analysis
# 
# The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. The data is widely used for collaborative filtering and other filtering solutions. However, we will be using this data to act as a means to demonstrate our skill in using Python to “play” with data.
# 
# #### Domain 
# Internet and Entertainment
# 
# **Note that the project will need you to apply the concepts of groupby and merging extensively.**

# #### 1. Import the necessary packages - 2.5 marks

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# In[3]:


pd.set_option('display.max_columns', None)


# #### 2. Read the 3 datasets into dataframes - 2.5 marks

# In[4]:


data = pd.read_csv("Data.csv")
item = pd.read_csv("item.csv")
user = pd.read_csv("user.csv")


# #### 3. Apply info, shape, describe, and find the number of missing values in the data - 5 marks

# In[5]:


data.info()
item.info()
user.info()


# In[6]:


data.shape


# In[7]:


item.shape


# In[8]:


user.shape


# In[9]:


data.describe()


# In[10]:


item.describe()


# In[11]:


user.describe()


# In[12]:


data.isnull().sum().sum()


# In[13]:


item.isnull().sum().sum()


# In[14]:


user.isnull().sum().sum()


# #### 4. Find the number of movies per genre using the item data - 5 marks

# In[15]:


# use sum on the default axis
item.sum(axis=0)


# #### 5. Find the movies that have more than one genre - 2.5 marks

# In[16]:


item.drop('movie id', axis=1).sum(axis=1, skipna = True) > 1


# #### 6. Drop the movie where the genre is unknown - 2.5 marks

# In[17]:


item1 = item[item.unknown == 0]


# In[18]:


item1.sum(axis=0)


# ### 7. Univariate plots of columns: 'rating', 'Age', 'release year', 'Gender' and 'Occupation' - 10 marks

# In[19]:


# HINT: use distplot for age and countplot for gender,ratings,occupation.
# HINT: Please refer to the below snippet to understand how to get to release year from release date. You can use str.split()
# as depicted below
# Hint : Use displot without kde for release year or line plot showing year wise count.


# In[20]:


a = 'My*cat*is*brown'
print(a.split('*')[3])

#similarly, the release year needs to be taken out from release date

#also you can simply slice existing string to get the desired data, if we want to take out the colour of the cat

print(a[10:])
print(a[-5:])


# In[21]:


#your answers here


# In[22]:


sns.distplot(user['age'], kde=False)
plt.show()


# In[23]:


Release = []
for i in range (len(item['release date'].values.tolist())):
    Release.append(item['release date'].values.tolist()[i][-4:])
    
    
sns.countplot(Release);


# In[24]:


sns.countplot(user['gender'])
plt.show()


# In[25]:


sns.countplot(user['occupation'])
plt.show()


# In[26]:


sns.distplot(Release, kde=False)


# ### 8. Visualize how popularity of genres has changed over the years - 10 marks
# 
# Note that you need to use the number of releases in a year as a parameter of popularity of a genre

# Hint 
# 
# 1: you need to reach to a data frame where the release year is the index and the genre is the column names (one cell shows the number of release in a year in one genre) or vice versa.
# Once that is achieved, you can either use univariate plots or can use the heatmap to visualise all the changes over the years 
# in one go. 
# 
# Hint 2: Use groupby on the relevant column and use sum() on the same to find out the nuumber of releases in a year/genre.  

# In[27]:


#Your answer here


# In[28]:


Release = list(map(int, Release))
Release_array = np.array(Release)


# In[29]:


item2 = item.drop(["movie id", "movie title", "release date"], axis=1)


# In[30]:


item2["release year"] = Release_array


# In[31]:


Trnd = item2.groupby("release year").sum()


# In[32]:


sns.lineplot(data = Trnd["unknown"])
sns.lineplot(data = Trnd["Action"])
sns.lineplot(data = Trnd["Adventure"])
sns.lineplot(data = Trnd["Animation"])
sns.lineplot(data = Trnd["Childrens"])
sns.lineplot(data = Trnd["Comedy"])
sns.lineplot(data = Trnd["Crime"])
sns.lineplot(data = Trnd["Documentary"])
sns.lineplot(data = Trnd["Drama"])
sns.lineplot(data = Trnd["Fantasy"])
sns.lineplot(data = Trnd["Film-Noir"])
sns.lineplot(data = Trnd["Horror"])
sns.lineplot(data = Trnd["Musical"])
sns.lineplot(data = Trnd["Mystery"])


# In[33]:


sns.heatmap(Trnd)


# In[34]:


#We can do this for all of other genres
sns.lineplot(data = Trnd["Mystery"])


# ### 9. Find the top 25 movies according to average ratings such that each movie has number of ratings more than 100 - 10 marks
# 
# Hint : 
# 1. First find the movies that have more than 100 ratings(use merge, groupby and count). Extract the movie titles in a list.
# 2. Find the average rating of all the movies and sort them in the descending order. You will have to use the .merge() function to reach to a data set through which you can get the names and the average rating.
# 3. Use isin(list obtained from 1) to filter out the movies which have more than 100 ratings.
# 
# Note: This question will need you to research about groupby and apply your findings. You can find more on groupby on https://realpython.com/pandas-groupby/.

# In[35]:


#your answer here


# In[96]:


top = pd.merge(data, item, on="movie id", how='inner')
top.shape


# In[86]:


top_list = top.groupby("movie title").rating.count()


# In[87]:


top_list_100 = []
for i in range (len(top_list)):
    if top_list[i] > 100:
        top_list_100.append([top_list.keys()[i], top_list[i]])
        
top_list_100


# In[88]:


top_rate = top.groupby("movie title").rating.mean()
top_rate_ds = top_rate.sort_values(ascending=False)
top_rate_ds


# In[89]:


top_list.isin(range(100,1000))


# ### 10. See gender distribution across different genres check for the validity of the below statements - 10 marks
# 
# * Men watch more drama than women
# * Women watch more Sci-Fi than men
# * Men watch more Romance than women
# 

# 1. There is no need to conduct statistical tests around this. Just compare the percentages and comment on the validity of the above statements.
# 
# 2. you might want ot use the .sum(), .div() function here.
# 3. Use number of ratings to validate the numbers. For example, if out of 4000 ratings received by women, 3000 are for drama, we will assume that 75% of the women watch drama.

# In[126]:


all_data = pd.merge(top, user, on="user id",  how='inner')


# In[106]:


Drama = all_data.groupby("gender").Drama.sum() 
Drama_W = Drama[0] #Number of total views of women who watched Drama Movies
Drama_M = Drama[1] #Number of total views of men who watched Drama Movies


# In[105]:


total_people = all_data.groupby("gender")['user id'].count()
total_women = total_people[0] #Number of total views of women
total_men = total_people[1] #Number of total views of men


# In[109]:


rate_of_drama_women = Drama_W / total_women #rate of wemen see drama movies
rate_of_drama_men = Drama_M / total_men #rate of men see drama movies


# In[113]:


rate_of_drama_women


# In[114]:


rate_of_drama_men


# In[116]:


Sci_Fi = all_data.groupby('gender')['Sci-Fi'].sum() 
Sci_Fi_W = Sci_Fi[0] #Number of total views of women who watched Sci_Fi Movies
Sci_Fi_M = Sci_Fi[1] #Number of total views of men who watched Sci_Fi Movies

rate_of_Sci_Fi_women = Sci_Fi_W / total_women #rate of wemen see Sci_Fi movies
rate_of_Sci_Fi_men = Sci_Fi_M / total_men #rate of men see Sci_Fi movies


# In[117]:


rate_of_Sci_Fi_women


# In[118]:


rate_of_Sci_Fi_men


# In[123]:


Romance = all_data.groupby('gender')['Romance'].sum() 
Romance_W = Romance[0] #Number of total views of women who watched Romance Movies
Romance_M = Romance[1] #Number of total views of men who watched Romance Movies

rate_of_Romance_women =Romance_W / total_women #rate of wemen see Romance movies
rate_of_Romance_men = Romance_M / total_men #rate of men see Romance movies


# In[124]:


rate_of_Romance_women


# In[125]:


rate_of_Romance_men


# #### Conclusion:
# 
# 

# 1- Women see drama movies more than men, since 42.8% of all women's views are considered to be drama movies. Where men has only 38.9% of all of their views are considered to be drama movies. 
# 2- Women and men see almost the same amount of Sci_Fi movies. Since women look at Sci_Fi movies around 10% of all time and men around 13.6%
# 3- Women see Romance movies more than men, since 22.8% of all women's views are considered to be Romance movies. Where men has only 18.3% of all of their views are considered to be Romance movies. 
