"""

NAMES:      Matt Brown, Makayla Townsend, Sam Ward
COURSE:     MIS 541-001
ASSIGNMENT: Group Project

Sections:
    0. IMPORTS
        * Importing packages needed for program functionality
    1. LOADING THE DATA
        * Reading the CSV file into a dataframe
    2. CLEANING THE DATA
        * Imputing missing or null values
    3. DISTRIBUTION EXPLORATION
        * Examining the distribution of major variables
    4. CORRELATION MATRIX
        * Generating a correlation matrix to find which variables relate to price
    5. LINEAR REGRESSION
        * Modeling price using linear regression (may want to use a variable selection technique?)
    6. UBER VS. LYFT LINEAR REGRESSION
        * Creating price regression models for Uber and Lyft separately and comparing them
    7. PRICE MAP
        * Using latitude and longitude to create a map of ride prices
    8. Maybe something with real estate prices? If we found a dataset of Boston prices to overlay with rideshare
    
Questions we're answering:
    What are the most important predictors for rideshare prices?
    Do the predictors vary for Uber vs. Lyft?
    How does rideshare price vary in different parts of Boston?
    Do the rideshare prices relate to real estate prices?
    Maybe something with weather and pricing/demand for rideshares?
    Do people use rideshares more when it's rainy/sunny/etc.?

    business impact aspects:
        * should you prepare more drivers based on the weather forecast?
        * consumer and business side (things as a user and uber/lyft)
        
    timeline: due week of the 6th

"""

"""
    0. IMPORTS
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, logit

"""
    1. LOADING THE DATA
"""

# load the file
df = pd.read_csv('rideshare_kaggle.csv')

# early exploration
print(df[['price', 'source', 'destination', 'cab_type']])
print(df.columns)

print(df[['short_summary', 'long_summary']])
print(df['short_summary'].value_counts())

"""
    2. CLEANING THE DATA
"""

# checking to see if any columns have null values: only price has missing values
print(df.isnull().any())

# 55,095 price values are missing
print(f"Number of missing price values: {df['price'].isnull().sum()}")

# there are 693,071 records total
print(f'Number of records: {len(df)}')

# that's less than 8%
print(f"Percent missing price: {df['price'].isnull().sum() / len(df)}")

# because of the small percent missing, we decided to remove them from the dataset
df = df.dropna()

# double-checking to verify: df now has no missing values
print(df.isnull().any())

# removing id and timestamp (all in eastern)
df = df.drop(columns=['id', 'timestamp'])

"""
    3. DISTRIBUTION EXPLORATION
"""

# printing the value counts for each column to double check that no data is missing but not NaN (like using 'missing' to indicate)
# at a glance everything looks clear
for column in df.columns:
    print(df[column].value_counts())

# next up (and this should maybe go in section 2) is removing columns we don't need and converting date/time stuff to datetime objects
# going to determine which we don't need with a correlation matrix

# histograms of some of the qualitative variables

# product id? i think this is the type of uber/lyft; idk what name is in comparison
print(df['product_id'].value_counts())
print(df['name'].value_counts())

# product id has some... hard to decipher guid values; name does not and they seem substitutable
print(df[['product_id', 'name']].value_counts())
# on second glance, product_id has uber values as guids, but name is fixed for everything
# we'll just use name then
df = df.drop('product_id', axis=1)

# making a histogram of names
name_df = df[['name', 'is_uber']].value_counts().to_frame()
name_df = name_df.reset_index()
name_df = name_df.set_index('name')

# as i do this, it occurs to me... these are all the same frequency? or very similar
# that's really odd to me, I'm wondering if this is a sample with control for product type

# going to skip a useless histogram for now, moving onto making a chart of average price
name_avg_price_df = df.groupby(['name']).mean()['price']

# this seems useful
plt.figure()
name_avg_price_df.plot(kind='bar')
plt.show()

# pick up with more of these: categorical variable price charts

"""
    4. CORRELATION MATRIX
"""

# making a flag field for cab type
df['is_uber'] = df['cab_type'].apply(lambda x: 1 if x == 'Uber' else 0)

# getting correlation matrix of variables to price
price_correlation_df = df.corr()['price'].to_frame()

# renaming column
price_correlation_df.columns = ['correlation']

# because we care about abs value, making a column to preserve sign
price_correlation_df['relationship'] = price_correlation_df['correlation'].apply(lambda x: 'direct' if x >= 0 else 'inverse')

# removing sign from correlation
price_correlation_df['correlation'] = price_correlation_df['correlation'].apply(lambda x: abs(x))

# sorting descending by correlation
price_correlation_df = price_correlation_df.sort_values(['correlation'], ascending=False)

# no reason to preserve price since it's the variable
price_correlation_df = price_correlation_df[1:]

# printing the result and plotting
print(price_correlation_df)

# maybe add color code relationship to this? honestly the takeaway is distance and surge are the only real numeric ones that matter
plt.figure()
price_correlation_df.plot(kind='bar')
plt.show()

"""
    5. LINEAR REGRESSION
"""

# making a model with distance, surge, and both
distance_model = ols("price ~ distance", df).fit()
surge_model = ols("price ~ surge_multiplier", df).fit()
combo_model = ols("price ~ distance + surge_multiplier", df).fit()

# examining summaries of all models
print(distance_model.summary())
print(surge_model.summary())
print(combo_model.summary())

# update: appears uber vs. lyft is notable (with uber being more expensive)
cab_type_model = ols("price ~ is_uber", df).fit()
print(cab_type_model.summary())

# making a full regression model with cab
model_with_cab = ols('price ~ distance + surge_multiplier + is_uber', df).fit()
print(model_with_cab.summary())

# very small bump to r-squared...

"""
    6. UBER VS. LYFT LINEAR REGRESSION
"""

# code here

"""
    7. PRICE MAP
"""

# code here

"""
    8. something with real estate prices?
"""

# code here

