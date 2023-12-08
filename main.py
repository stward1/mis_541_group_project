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
from statsmodels.formula.api import ols
import statistics

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

# pulling in average rent for each neighborhood via rentcafe: https://www.rentcafe.com/average-rent-market-trends/us/ma/boston/
rent_df = pd.read_csv('boston_neighborhood_rentcafe.csv')

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

# making a flag field for cab type
df['is_uber'] = df['cab_type'].apply(lambda x: 1 if x == 'Uber' else 0)

# what about price per mile as a metric
df['price_dist_ratio'] = df['price'] / df['distance']


"""
    3. DISTRIBUTION EXPLORATION
"""

# for later analysis, splitting uber and lyft frames
uber_frame = df[df['is_uber'] == 1]
lyft_frame = df[df['is_uber'] == 0]

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
# maybe do [price]mean, i think it's faster?
name_avg_price_df = df.groupby(['name'])['price'].mean()
uber_name_df = uber_frame.groupby(['name'])['price'].mean()
lyft_name_df = lyft_frame.groupby(['name'])['price'].mean()

# this seems useful: average price by type
plt.figure()
lyft_name_df.plot(kind='bar', color='red')
plt.title('Average Price by Ride Type (Lyft)')
plt.ylabel('Average Price (USD)')
plt.xlabel('Ride Type')
plt.xticks(rotation=45)
plt.show()

plt.figure()
uber_name_df.plot(kind='bar', color='green')
plt.title('Average Price by Ride Type (Uber)')
plt.ylabel('Average Price (USD)')
plt.xlabel('Ride Type')
plt.xticks(rotation=45)
plt.show()

# think it would be cool to compare average price to total revenue (sum of price)
# this is... not useful because there are about the same number of each type
# which i noticed earlier... oops lol
uber_name_df = uber_frame.groupby(['name'])['price'].sum()
lyft_name_df = lyft_frame.groupby(['name'])['price'].sum()

# this seems useful: total price (revenue) by type
plt.figure()
lyft_name_df.plot(kind='bar', color='red')
plt.title('Total Revenue by Ride Type (Lyft)')
plt.ylabel('Revenue (USD)')
plt.xlabel('Ride Type')
plt.xticks(rotation=45)
plt.show()

plt.figure()
uber_name_df.plot(kind='bar', color='green')
plt.title('Total Revenue by Ride Type (Uber)')
plt.ylabel('Revenue (USD)')
plt.xlabel('Ride Type')
plt.xticks(rotation=45)
plt.show()



# weather average price: literally no effect, like at all lol
weather_avg_price_df = df.groupby(['short_summary'])['price'].mean()
weather_avg_price_uber = uber_frame.groupby(['short_summary'])['price'].mean()
weather_avg_price_lyft = lyft_frame.groupby(['short_summary'])['price'].mean()
plt.figure()
weather_avg_price_df.plot(kind='line')
weather_avg_price_uber.plot(kind='line', color='green')
weather_avg_price_lyft.plot(kind='line', color='red')
plt.title('Average Price by Weather')
plt.ylabel('Average Price (USD)')
plt.xlabel('Weather')
plt.xticks(rotation=45)
plt.show()

weather_avg_price_df = df.groupby(['short_summary'])['distance'].mean()
plt.figure()
weather_avg_price_df.plot(kind='bar')
plt.show()

# how about by hour of the day
hour_avg_price_df = df.groupby(['hour'])['price'].mean()
plt.figure()
hour_avg_price_df.plot(kind='line')
plt.show()





uber_name_df = uber_frame.groupby(['name'])['price'].sum()
lyft_name_df = lyft_frame.groupby(['name'])['price'].sum()


# now *this* is a little more interesting: the price per mile spikes around rush hours and midday
hour_avg_price_ratio_df = df.groupby(['hour'])['price_dist_ratio'].mean()
hour_avg_price_ratio_uber = uber_frame.groupby(['hour'])['price_dist_ratio'].mean()
hour_avg_price_ratio_lyft = lyft_frame.groupby(['hour'])['price_dist_ratio'].mean()

# this is examining the price/distance ratio, which is think is dollars and miles
plt.figure()
hour_avg_price_ratio_df.plot(kind='line')
hour_avg_price_ratio_uber.plot(kind='line', color='green')
hour_avg_price_ratio_lyft.plot(kind='line', color='red')
plt.title('Average Price-Distance Ratio by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Price-Distance Ratio (USD per Mile)')
plt.show()

# surge mult: really spikes around lunch
hour_avg_surge_df = df.groupby(['hour'])['surge_multiplier'].mean()
uber_avg_surge_df = uber_frame.groupby(['hour'])['surge_multiplier'].mean()
lyft_avg_surge_df = lyft_frame.groupby(['hour'])['surge_multiplier'].mean()

plt.figure()
# hour_avg_surge_df.plot(kind='line')
# uber_avg_surge_df.plot(kind='line', color='green')
lyft_avg_surge_df.plot(kind='line', color='red')
plt.title('Average Surge Multiplier by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Surge Multiplier')
plt.show()

# uber doesn't use surge multiplier


# pick up with more of these: categorical variable price charts

"""
    4. CORRELATION MATRIX
"""


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

# pruning out price_dist_ratio, that's not meant to be in there, it uses price
price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)

# really only the top 5 or so matter
price_correlation_df = price_correlation_df[0:5]

price_correlation_df = price_correlation_df.reset_index()
price_correlation_df.columns = ['variable', 'correlation', 'relationship']

# printing the result and plotting
print(price_correlation_df)

# maybe add color code relationship to this? honestly the takeaway is distance and surge are the only real numeric ones that matter
plt.figure()
plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'blue','inverse':'black'}))
plt.xticks(rotation=45)
plt.xlabel('Variable')
plt.ylabel('Correlation Strength')
plt.title('Overall Variable Correlations (Top 5)')
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

# real estate modeling
estate_model = ols("price ~ distance + surge_multiplier + source_rent + destination_rent", df).fit()
print(estate_model.summary())


"""
    6. UBER VS. LYFT LINEAR REGRESSION
"""

# code here
# do this tonight

"""
    6a. Uber
"""


price_correlation_df = uber_frame.corr()['price'].to_frame()

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

# pruning out price_dist_ratio, that's not meant to be in there, it uses price
price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)

# really only the top 5 or so matter
price_correlation_df = price_correlation_df[0:5]

price_correlation_df = price_correlation_df.reset_index()
price_correlation_df.columns = ['variable', 'correlation', 'relationship']

# printing the result and plotting
print(price_correlation_df)

# maybe add color code relationship to this? honestly the takeaway is distance and surge are the only real numeric ones that matter
plt.figure()
plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'green','inverse':'black'}))
plt.xticks(rotation=45)
plt.xlabel('Variable')
plt.ylabel('Correlation Strength')
plt.title('Uber Variable Correlations (Top 5)')
plt.show()


distance_model = ols("price ~ distance", uber_frame).fit()

# examining summaries of all models
print(distance_model.summary())

"""
    6b. Lyft
"""

price_correlation_df = lyft_frame.corr()['price'].to_frame()

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

# pruning out price_dist_ratio, that's not meant to be in there, it uses price
price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)

# really only the top 5 or so matter
price_correlation_df = price_correlation_df[0:5]

price_correlation_df = price_correlation_df.reset_index()
price_correlation_df.columns = ['variable', 'correlation', 'relationship']

# printing the result and plotting
print(price_correlation_df)

# maybe add color code relationship to this? honestly the takeaway is distance and surge are the only real numeric ones that matter
plt.figure()
plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'red','inverse':'black'}))
plt.xticks(rotation=45)
plt.xlabel('Variable')
plt.ylabel('Correlation Strength')
plt.title('Lyft Variable Correlations (Top 5)')
plt.show()


distance_model = ols("price ~ distance", lyft_frame).fit()
surge_model = ols("price ~ surge_multiplier", lyft_frame).fit()
combo_model = ols("price ~ distance + surge_multiplier", lyft_frame).fit()

# examining summaries of all models
print(distance_model.summary())
print(surge_model.summary())
print(combo_model.summary())

"""
    7. PRICE MAP
"""

# code here
# maybe?

# graphing rents
# restaurant_review_frame.join(other=restaurant_ids_dataframe,on='business_id',how='left')

# df.join(other=rent_df, on)

# getting a list of all the neighborhoods in the df and from the rent-cafe data
source_neighborhoods = list(df['source'].unique())
source_neighborhoods.sort()
destination_neighborhoods = list(df['destination'].unique())
destination_neighborhoods.sort()
rent_neighborhoods = list(rent_df['neighborhood'])
rent_neighborhoods.sort()

# printing each out to see what needs to be renamed/removed
print(source_neighborhoods)
print(destination_neighborhoods)
print(rent_neighborhoods)

blank_col = [0] * len(rent_df)
rent_df['isKept'] = blank_col
rent_df = rent_df.set_index(['neighborhood'])


# manually looking through results and figuring out how to rename
rent_df.loc['Back Bay', ['isKept']] = 1
rent_df.loc['Beacon Hill', ['isKept']] = 1

# Boston University missing, as per googling will make this Fenway
rent_df.loc['Boston University'] = rent_df.loc['Fenway - Kenmore']
rent_df.loc['Boston University', ['isKept']] = 1

# Fenway - Kenmore becomes Fenway
rent_df.loc['Fenway'] = rent_df.loc['Fenway - Kenmore']
rent_df.loc['Fenway', ['isKept']] = 1

# Downtown - Financial District becomes Financial District
rent_df.loc['Financial District'] = rent_df.loc['Downtown - Financial District']
rent_df.loc['Financial District', ['isKept']] = 1

# haymarket square is between west and south end? going to use average of south/west end rent
rent_df.loc['Haymarket Square'] = rent_df.loc['South End Boston']
south_west_avg = statistics.mean([rent_df.loc['South End Boston']['average_rent'], rent_df.loc['West End Boston']['average_rent']])
rent_df.loc['Haymarket Square', ['average_rent']] = south_west_avg
rent_df.loc['Haymarket Square', ['isKept']] = 1

# north end boston becomes north end
rent_df.loc['North End'] = rent_df.loc['North End Boston']
rent_df.loc['North End', ['isKept']] = 1

# north station is between north/west end, doing average
rent_df.loc['North Station'] = rent_df.loc['North End Boston']
north_west_avg = statistics.mean([rent_df.loc['North End Boston']['average_rent'], rent_df.loc['West End Boston']['average_rent']])
rent_df.loc['North Station', ['average_rent']] = north_west_avg
rent_df.loc['North Station', ['isKept']] = 1

# northeastern university is in fenway - kenmore (already renamed to fenway and kept)
rent_df.loc['Northeastern University'] = rent_df.loc['Fenway']

# south station is downtown, which was renamed to financial district
rent_df.loc['South Station'] = rent_df.loc['Financial District']

# theatre district is bay village
rent_df.loc['Theatre District'] = rent_df.loc['Bay Village']
rent_df.loc['Theatre District', ['isKept']] = 1

# west end boston becomes west end
rent_df.loc['West End'] = rent_df.loc['West End Boston']
rent_df.loc['West End', ['isKept']] = 1

cleaned_rent_df = rent_df[rent_df['isKept'] == 1]
cleaned_rent_df = cleaned_rent_df.drop(['isKept'], axis=1)

cleaned_rent_df = cleaned_rent_df.set_index(['neighborhood'])
# plotting quick bar chart... this wasn't as dramatic as i was hoping it would be
plt.figure()
cleaned_rent_df.plot(kind='bar')
plt.xticks(rotation=45)
plt.title('Average Rent by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rent (USD)')
plt.show()


cleaned_rent_df = cleaned_rent_df.reset_index()
# joining back to the main df
df = pd.merge(df, cleaned_rent_df, how='left', left_on='source', right_on='neighborhood')
df = df.rename(columns={'average_rent': 'source_rent'})
df = df.drop(['neighborhood'], axis=1)
# doing the same for destination
df = pd.merge(df, cleaned_rent_df, how='left', left_on='destination', right_on='neighborhood')
df = df.rename(columns={'average_rent': 'destination_rent'})
df = df.drop(['neighborhood'], axis=1)




"""
    8. something with real estate prices?
"""

# making a map of the prices, very disappointing
# use c = price bin and make it x / mean price to have a nice gradient instead
plt.figure()
plt.scatter(df['longitude'], df['latitude'], s=1, c=df['price'], alpha=0.5)
plt.title('Attempted Price Map of Boston')
plt.xlabel('Longitude')
plt.xlabel('Latitude')
plt.show()


# what about average price of the different neighborhoods?
# source average price frame
source_price_df = df.groupby(['source'])['price'].mean().to_frame()
source_price_df.columns = ['avg_source_price']

# destination average price frame
destination_price_df = df.groupby(['destination'])['price'].mean().to_frame()
destination_price_df.columns = ['avg_destination_price']

# adding in rent price
rent_price_df = cleaned_rent_df
rent_price_df = rent_price_df.reset_index()
rent_price_df.columns = ['source', 'avg_rent_thousands']
rent_price_df = rent_price_df.set_index(['source'])

# joining on index
neighborhood_df = source_price_df.merge(destination_price_df, how='left', left_index=True, right_index=True)
neighborhood_df = neighborhood_df.merge(rent_price_df, how='left', left_index=True, right_index=True)
neighborhood_df['avg_rent_thousands'] = neighborhood_df['avg_rent_thousands'] / 1000

# weak relationship between rent and rideshare prices
plt.figure()
neighborhood_df.plot(kind='bar')
plt.title('Average Pricing and Rent by Neighborhood')
plt.xlabel('Neighborhood')
plt.xticks(fontsize=8)
plt.ylabel('Average Price (USD) or Rent (thousands USD)')
plt.show()

# modest inverse correlation
print(neighborhood_df.corr().to_string())


# code here

