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

# code here
# get this done shortly to open up later exploration

"""
    3. DISTRIBUTION EXPLORATION
"""

# code here

"""
    4. CORRELATION MATRIX
"""

# code here

"""
    5. LINEAR REGRESSION
"""

# code here

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

