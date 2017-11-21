
# coding: utf-8

# In[187]:

# Importing Pandas，Numpy and Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# # Question Phase

# For this project I come up one question that I'll address in following analysis：
# 
# Q: What's the difference(characteristics) of those people who survived against those didn't?

# # Data Wrangling Phase

# In[188]:

# Load the data into notebook and show the first 5 records for glimpse
df = pd.read_csv('titanic-data.csv')
df.head()


# In[51]:

# Take a look at some attributes information
df.info()


# From above we can see that ticket column is irrelevant to our intention, and cabin column contains so many missing value that it hardly contribute to our final goal (Age column also has the same issue but not that many). So for this project I'd like to take out ticket and cabin column. 
# 
# As for those missing values in the data, I plan to use the median of Age to fill those NAs in Age column for the ease of following analysis. And just ignore those 2 NAs in Embarked since there is no way to say which location those 2 people embarked upon the ship.

# In[189]:

# Execute data cleansing and check
df.drop(['Ticket','Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df.head()


# # Explore & Analysis Phase

# I believe the number of women and children(age under 18) who survived is greater than the number of men since women and children should be allowed to escape first for the common practice in accidents. 
# 
# And I think the number of survivors in 1st class should be more than those in 2nd and 3rd class respectively since 1st class cabin is closer to the deck.

# In[190]:

# df now only contains those records with value in Age column! Below codes gives the descriptive statistic result 
# to get an overview of the processed data
df.loc[:,['Age', 'SibSp', 'Parch', 'Fare']][df.Survived==1].describe()


# In[191]:

df.loc[:,['Age', 'SibSp', 'Parch', 'Fare']][df.Survived==0].describe()


# From above we can tell those survivors are about 2 years younger than victims on average, and they spen more for tickets (maybe they were in the 1st class. We'll check about that later), and those who with less relatives are more likely to survive.
# 
# I'll start checking on Sex, Age, Pclass, SibSp, and Parch against survival data (as well as cross check) to tell whether these features affect the odds of survival.

# Now let's look into the 1D visualizations for Survived, Pclass, and Sex respectively

# In[196]:

sns.countplot(x="Survived", data=df)
sns.plt.title('Survived Data (1 means survive)')


# Number of victims is much higher than survivors

# In[199]:

sns.countplot(x="Pclass", data=df)
sns.plt.title('Count in terms of Pclass')


# More people in class than the total of class 1 and 2

# In[198]:

sns.countplot(x="Sex", data=df)
sns.plt.title('Count in terms of Sex')


# In[211]:

A lot more men on board than women


# In[161]:

# To plot the survival data versus Sex
Survived_sex = pd.crosstab(df.Sex, df.Survived)
Survived_sex.rename(columns={0:'Dead',1:'Survived'},inplace=True)
# Add the y label back 
Survived_sex.plot.bar(figsize=(5,5)).set_ylabel("Number of People")
plt.title('Survival Comparision between Sex')


# As we can see from above chart, there were a lot more women than men who survived from this tragedy, that may because the "lady and children go first" rule in tragedy.

# In[162]:

# To plot the survival data versus Age in order to look into the age difference between survivors and victims
Survived_age_pclass = pd.crosstab(df.Age, df.Survived)
Survived_age_pclass.rename(columns={0:'Dead',1:'Survived'},inplace=True)
Survived_age_pclass.plot.line(figsize=(15,6)).set_ylabel("Number of People")
plt.title('Survival over age')


# It seems people whose age between 20 and 30 are less likely to survive, let's look deeper into the data.

# In[91]:

# Base on the above chart, try to get a summary from those victims who age between 20 and 30 where people are less
# likely to survive
df.loc[:,['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']][(df['Age'] >= 20) & (df['Age'] <= 30) & (df['Survived']==0)].describe()


# In[164]:

# Plot the survival data against selected group to break down the data
Survived_age_class = pd.crosstab(df.Pclass[(df['Age'] >= 20) & (df['Age'] <= 30)], df.Survived[(df['Age'] >= 20) & (df['Age'] <= 30)])
Survived_age_class.rename(columns={0:'Dead',1:'Survived'},inplace=True)
# Add rot parameter for better reading
Survived_age_class.plot.bar(figsize=(10,6), rot=0).set_ylabel("Number of People")
# Add some explanation
plt.title('Survival among Pclass (Only include passengers age between 20 and 30)')


# Seen from the above table and charts, people are less likely to survive not only when their age between 20 and 30, but also when they were in class 3. A simple guess is there were a lot of young immigrants who can only afford a class 3 ticket, trying to get to the state side(This is merely a hypothesis, an additional survey is required to prove it but it beyonds the scope of this project).
# 
# Next I'd like to check the survival rate on children, and since there isn't a column states whether someone is children(age under 18) or not, we need a function to tell if a record was a child or not. And we need a new column to store the information

# In[201]:

def Child_Classifier(age):
    # return Children if the person is a child, and Adult otherwise
    if age < 18:
        return 'Children'
    else:
        return 'Adult'


# In[202]:

# Create a new column to store the info
df['Child'] = df['Age'].apply(Child_Classifier) 
df.head()


# Next I'd like to check

# In[203]:

# Plot the survival data against selected group to break down the data
Children_Sur = pd.crosstab(df.Child, df.Survived)
Children_Sur.rename(columns={0:'Dead',1:'Survived'},inplace=True)
Children_Sur.plot.bar(figsize=(10,5), rot=0).set_ylabel("Number of People")
plt.title('Children Survival Data')


# The casualty rate for Adult is much higher than Children group

# In[125]:

# Now we can check the survival rate over class in terms of whether someone is child, this time I chose to use
# another way to plot the data for simplicity
sns.factorplot('Pclass',data=df[df.Survived==0],hue='Child',kind='count')


# From above we can know that children victims are from class 3

# In[204]:

def Alone_Passenger(family):
    no_of_sipsp, no_of_parch = family
    no_of_relatives = no_of_sipsp + no_of_parch
    # Return 'Alone' if alone, 'Have relatives' otherwise
    if no_of_relatives == 0:
        return 'Alone'
    else:
        return 'Have relatives'


# In[205]:

# Create a new column to store the alone info, and this time we need to specify axis for apply since there're
# 2 columns involved
df['Alone'] = df[['SibSp','Parch']].apply(Alone_Passenger, axis = 1)
df.head()


# In[207]:

Alone_Sur = pd.crosstab(df.Alone, df.Survived)
Alone_Sur.rename(columns={0:'Dead',1:'Survived'},inplace=True)
Alone_Sur.plot.bar(figsize=(10,6),rot=0).set_ylabel("Number of People")
plt.title('Family Survival Data')


# In[210]:

Class_Alone_Sur = pd.crosstab(df.Alone, df.Pclass)
#Children_Sur.rename(columns={0:'Dead',1:'Survived'},inplace=True)
Class_Alone_Sur.plot.bar(figsize=(10,6),rot=0).set_ylabel("Number of People")
plt.title('Alone VS Family')


# We can know from above chart that a lot more alone passengers (without any relative on board the ship) lost their lives than those with relative(s), whilst this difference is not that big for survivors.

# Code Reference for the project:
# 
# 1. http://pandas.pydata.org/pandas-docs/stable/10min.html#getting
# 2. https://stackoverflow.com/questions/15315452/selecting-with-complex-criteria-from-pandas-dataframe
# 3. http://www.cnblogs.com/jasonfreak/p/5441512.html
# 4. http://pandas.pydata.org/pandas-docs/stable/missing_data.html
# 5. http://pandas.pydata.org/pandas-docs/stable/visualization.html

# # Conclusion

# As we can see from above analysis, more women than men survived in this tragedy. And people who age at around 20 and 30 have a lower survival rate than other age groups.
# 
# Children in class 3 also have a higher casualty than those in class 1 and 2, and the same for those people who traveled alone (versus those men with relative(s). The analysis reveals that most of them are in class 3， it might because young immigrants often travel alone.
# 
# Due to some demography info not provided in this dataset, some further analysis are yet to be performed. My analysis only look into some simple statistic into the data. I would like to use logistic regression to build a model and see what features affect the survival rate and try to predict what kind of passenger is more likely to survive.

# In[ ]:



