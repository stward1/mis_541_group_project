"""

NAMES:      Matt Brown, Makayla Townsend, Sam Ward
COURSE:     MIS 541-001
ASSIGNMENT: Group Project

Sections:
    0. IMPORTS
        * Importing packages needed for program functionality
    1. LOADING THE DATA
        * Reading the CSV files into dataframes
    2. CLEANING THE DATA
        * Imputing missing or null values
    3. FEATURE CREATION
    4. DISTRIBUTION EXPLORATION
    5. GRAPHICAL ANALYSES
    6. CORRELATION MATRICES
    7. LINEAR REGRESSION
    8. MAIN FUNCTION
    9. MAIN CALL
        
        
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
import matplotlib.patches as patches
from statsmodels.formula.api import ols
import statistics

"""
    1. LOADING THE DATA
"""

# loading rideshare data via kaggle
def load_rideshare():
    return pd.read_csv('rideshare_kaggle.csv')

# loading rent data
def load_rents():
    # average rent for each neighborhood via rentcafe: https://www.rentcafe.com/average-rent-market-trends/us/ma/boston/
    return pd.read_csv('boston_neighborhood_rentcafe.csv')

"""
    2. CLEANING THE DATA
"""

# removing missing values from the rideshare data
def clean_rideshare(rideshare_df):
    print('________________________________________________________________\n')
    print("CLEANING RIDESHARE DATA:")
    
    # checking to see if any columns have null values: only price has missing values
    print(rideshare_df.isnull().any())
    
    # 55,095 price values are missing
    print(f"Number of missing price values: {rideshare_df['price'].isnull().sum()}")
    
    # there are 693,071 records total
    print(f'Number of records: {len(rideshare_df)}')
    
    # that's less than 8%
    print(f"Percent missing price: {rideshare_df['price'].isnull().sum() / len(rideshare_df)}")
    
    # because of the small percent missing, we decided to remove them from the dataset
    rideshare_df = rideshare_df.dropna()
    
    # double-checking to verify: df now has no missing values
    print(rideshare_df.isnull().any())
    
    # removing id and timestamp (all in eastern)
    rideshare_df = rideshare_df.drop(columns=['id', 'timestamp'])
    
    # removing product_id because name is a better alternative
    rideshare_df = rideshare_df.drop(columns=['product_id'])

    print("END RIDESHARE CLEANING")
    print('________________________________________________________________\n')
    return rideshare_df

# adjusting the rent data to align with the same neighborhoods in the rideshare frame
def clean_rent(rent_df, rideshare_df):
    print('________________________________________________________________\n')
    print("CLEANING RENT DATA:")
    
    # getting a list of all the neighborhoods in rideshare and from the rent-cafe data
    source_neighborhoods = list(rideshare_df['source'].unique())
    source_neighborhoods.sort()
    destination_neighborhoods = list(rideshare_df['destination'].unique())
    destination_neighborhoods.sort()
    rent_neighborhoods = list(rent_df['neighborhood'])
    rent_neighborhoods.sort()

    # printing each out to see what needs to be renamed/removed
    print(source_neighborhoods)
    print(destination_neighborhoods)
    print(rent_neighborhoods)

    # making a flag field set to 0 by default and making neighborhood index for rent_df
    blank_col = [0] * len(rent_df)
    rent_df['isKept'] = blank_col
    rent_df = rent_df.set_index(['neighborhood'])

    # manually looking through results and figuring out how to rename
    # these two are kept exactly the same
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
    
    # with everything adjusted, retain the rows to keep and drop the flag
    cleaned_rent_df = rent_df[rent_df['isKept'] == 1]
    cleaned_rent_df = cleaned_rent_df.drop(['isKept'], axis=1)
    
    print("END RENT CLEANING")
    print('________________________________________________________________\n')
    return cleaned_rent_df

"""
    3. FEATURE CREATION
"""

# adding a column for whether or not a rideshare was uber
def add_is_uber(rideshare_df):
    # making a flag field for cab type
    rideshare_df['is_uber'] = rideshare_df['cab_type'].apply(lambda x: 1 if x == 'Uber' else 0)
    return rideshare_df

# adding a column for the ratio of rideshare price to its distance
def add_price_dist_ratio(rideshare_df):
    # what about price per mile as a metric
    rideshare_df['price_dist_ratio'] = rideshare_df['price'] / rideshare_df['distance']
    return rideshare_df

# adding the average rent of the source/destination of a ride
def add_average_rents(rideshare_df, rent_df):
    # rent_df has neighborhood as index, here it needs to be a column
    join_rent_df = rent_df.reset_index()
    
    # joining back to the main df for source
    rideshare_df = pd.merge(rideshare_df, join_rent_df, how='left', left_on='source', right_on='neighborhood')
    rideshare_df = rideshare_df.rename(columns={'average_rent': 'source_rent'})
    rideshare_df = rideshare_df.drop(['neighborhood'], axis=1)
    
    # doing the same for destination
    rideshare_df = pd.merge(rideshare_df, join_rent_df, how='left', left_on='destination', right_on='neighborhood')
    rideshare_df = rideshare_df.rename(columns={'average_rent': 'destination_rent'})
    rideshare_df = rideshare_df.drop(['neighborhood'], axis=1)

    # returning full frame
    return rideshare_df

# returns dataframe of just the uber rides
def get_uber_frame(rideshare_df):
    return rideshare_df[rideshare_df['is_uber'] == 1]

# returns dataframe of just the lyft rides
def get_lyft_frame(rideshare_df):
    return rideshare_df[rideshare_df['is_uber'] == 0]

"""
    4. DISTRIBUTION EXPLORATION
"""

# displays early exploration of data
def prelim_exploration(rideshare_df):
    print('________________________________________________________________\n')
    print('PRELIMINARY EXPLORATION:')

    # looking at basic data and columns
    print(rideshare_df[['price', 'source', 'destination', 'cab_type']])
    print(rideshare_df.columns)
    
    # examining ride summaries (weather)
    print(rideshare_df[['short_summary', 'long_summary']])
    print(rideshare_df['short_summary'].value_counts())

    print('END PRELIMINARY EXPLORATION')
    print('________________________________________________________________\n')
    
# examines value counts for each column
def value_count_exploration(rideshare_df):
    print('________________________________________________________________\n')
    print('VALUE COUNT EXPLORATION:')
    
    # printing the value counts for each column
    for column in rideshare_df.columns:
        print(rideshare_df[column].value_counts())

    print('END VALUE COUNT EXPLORATION')
    print('________________________________________________________________\n')

# looks at the distribution of the different rideshare types (names)
def name_exploration(rideshare_df):
    print('________________________________________________________________\n')
    print('NAME EXPLORATION:')

    # discovering that the data has about even distributions of names
    print(rideshare_df[['name', 'is_uber']].value_counts().to_frame())

    print('END NAME EXPLORATION')
    print('________________________________________________________________\n')

"""
    5. GRAPHICAL ANALYSES
"""

# analyses the average price of each type of rideshare
def average_price_by_name(uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE PRICE BY NAME:')

    # making frames for uber and lyft that have average price aggregated by name
    uber_name_df = uber_frame.groupby(['name'])['price'].mean()
    lyft_name_df = lyft_frame.groupby(['name'])['price'].mean()    

    # displaying uber
    print('Uber:')
    print(uber_name_df)
    plt.figure()
    uber_name_df.plot(kind='bar', color='green')
    plt.title('Average Price by Ride Type (Uber)')
    plt.ylabel('Average Price (USD)')
    plt.xlabel('Ride Type')
    plt.xticks(rotation=45)
    plt.show()

    # displaying lyft
    print('Lyft:')
    print(lyft_name_df)
    plt.figure()
    lyft_name_df.plot(kind='bar', color='red')
    plt.title('Average Price by Ride Type (Lyft)')
    plt.ylabel('Average Price (USD)')
    plt.xlabel('Ride Type')
    plt.xticks(rotation=45)
    plt.show()

    print('END AVERAGE PRICE BY NAME')
    print('________________________________________________________________\n')

# analyses total price (revenue) by type of rideshare: this isn't useful because of the even sampling, but this was done to double check names
def total_price_by_name(uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('TOTAL PRICE (REVENUE) BY NAME:')

    # making frames for uber and lyft that have total price aggregated by name
    uber_name_df = uber_frame.groupby(['name'])['price'].sum()
    lyft_name_df = lyft_frame.groupby(['name'])['price'].sum()    

    # displaying uber
    print('Uber:')
    print(uber_name_df)
    plt.figure()
    uber_name_df.plot(kind='bar', color='green')
    plt.title('Total Revenue by Ride Type (Uber)')
    plt.ylabel('Revenue (USD)')
    plt.xlabel('Ride Type')
    plt.xticks(rotation=45)
    plt.show()

    # displaying lyft
    print('Lyft:')
    print(lyft_name_df)
    plt.figure()
    lyft_name_df.plot(kind='bar', color='red')
    plt.title('Total Revenue by Ride Type (Lyft)')
    plt.ylabel('Revenue (USD)')
    plt.xlabel('Ride Type')
    plt.xticks(rotation=45)
    plt.show()

    print('END TOTAL PRICE (REVENUE) BY NAME')
    print('________________________________________________________________\n')

# analyses average price by the weather... the effects turned out to be mild
def average_price_by_weather(rideshare_df, uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE PRICE BY WEATHER:')

    # making frames for overall/uber/lyft with average price agg by weather (short summary)
    weather_avg_price_df = rideshare_df.groupby(['short_summary'])['price'].mean()
    weather_avg_price_uber = uber_frame.groupby(['short_summary'])['price'].mean()
    weather_avg_price_lyft = lyft_frame.groupby(['short_summary'])['price'].mean()

    # printing results
    print('Overall:')
    print(weather_avg_price_df)
    print('Uber:')
    print(weather_avg_price_uber)
    print('Lyft:')
    print(weather_avg_price_lyft)
    
    # plotting multi line chart
    plt.figure()
    weather_avg_price_df.plot(kind='line', color='blue')
    weather_avg_price_uber.plot(kind='line', color='green')
    weather_avg_price_lyft.plot(kind='line', color='red')
    plt.title('Average Price by Weather')
    plt.ylabel('Average Price (USD)')
    plt.xlabel('Weather')
    overall_label = patches.Patch(color='blue', label='Overall')
    uber_label = patches.Patch(color='green', label='Uber')
    lyft_label = patches.Patch(color='red', label='Lyft')
    plt.legend(handles=[overall_label, uber_label, lyft_label], loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

    print('END AVERAGE PRICE BY WEATHER')
    print('________________________________________________________________\n')

# analyses average distance of a ride by weather
def average_distance_by_weather(rideshare_df, uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE DISTANCE BY WEATHER:')

    # making frames for overall/uber/lyft with average distance agg by weather (short summary)
    weather_avg_dist_df = rideshare_df.groupby(['short_summary'])['distance'].mean()
    weather_avg_dist_uber = uber_frame.groupby(['short_summary'])['distance'].mean()
    weather_avg_dist_lyft = lyft_frame.groupby(['short_summary'])['distance'].mean()

    # printing results
    print('Overall:')
    print(weather_avg_dist_df)
    print('Uber:')
    print(weather_avg_dist_uber)
    print('Lyft:')
    print(weather_avg_dist_lyft)
    
    # plotting multi line chart
    plt.figure()
    weather_avg_dist_df.plot(kind='line')
    weather_avg_dist_uber.plot(kind='line', color='green')
    weather_avg_dist_lyft.plot(kind='line', color='red')
    plt.title('Average Distance by Weather')
    plt.ylabel('Average Distance (miles)')
    plt.xlabel('Weather')
    overall_label = patches.Patch(color='blue', label='Overall')
    uber_label = patches.Patch(color='green', label='Uber')
    lyft_label = patches.Patch(color='red', label='Lyft')
    plt.legend(handles=[overall_label, uber_label, lyft_label], loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

    print('END AVERAGE DISTANCE BY WEATHER')
    print('________________________________________________________________\n')

# analyses average price of a ride by hour of the day
def average_price_by_hour(rideshare_df, uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE PRICE BY HOUR:')

    # making frames for overall/uber/lyft with average price agg by hour
    hour_avg_price_df = rideshare_df.groupby(['hour'])['price'].mean()
    hour_avg_price_uber = uber_frame.groupby(['hour'])['price'].mean()
    hour_avg_price_lyft = lyft_frame.groupby(['hour'])['price'].mean()

    # printing results
    print('Overall:')
    print(hour_avg_price_df)
    print('Uber:')
    print(hour_avg_price_uber)
    print('Lyft:')
    print(hour_avg_price_lyft)
    
    # plotting multi line chart
    plt.figure()
    hour_avg_price_df.plot(kind='line')
    hour_avg_price_uber.plot(kind='line', color='green')
    hour_avg_price_lyft.plot(kind='line', color='red')
    plt.title('Average Price by Hour')
    plt.ylabel('Average Price (USD)')
    plt.xlabel('Hour')
    overall_label = patches.Patch(color='blue', label='Overall')
    uber_label = patches.Patch(color='green', label='Uber')
    lyft_label = patches.Patch(color='red', label='Lyft')
    plt.legend(handles=[overall_label, uber_label, lyft_label], loc='upper right')
    plt.show()

    print('END AVERAGE PRICE BY HOUR')
    print('________________________________________________________________\n')

# analyses average price/dist ratio of a ride by hour of the day
def average_price_dist_ratio_by_hour(rideshare_df, uber_frame, lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE PRICE DISTANCE RATIO BY HOUR:')

    # making frames for overall/uber/lyft with average price dist agg by hour
    hour_avg_price_ratio_df = rideshare_df.groupby(['hour'])['price_dist_ratio'].mean()
    hour_avg_price_ratio_uber = uber_frame.groupby(['hour'])['price_dist_ratio'].mean()
    hour_avg_price_ratio_lyft = lyft_frame.groupby(['hour'])['price_dist_ratio'].mean()

    # printing results
    print('Overall:')
    print(hour_avg_price_ratio_df)
    print('Uber:')
    print(hour_avg_price_ratio_uber)
    print('Lyft:')
    print(hour_avg_price_ratio_lyft)
    
    # plotting multi line chart
    plt.figure()
    hour_avg_price_ratio_df.plot(kind='line')
    hour_avg_price_ratio_uber.plot(kind='line', color='green')
    hour_avg_price_ratio_lyft.plot(kind='line', color='red')
    plt.title('Average Price-Distance Ratio by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Price-Distance Ratio (USD per Mile)')
    overall_label = patches.Patch(color='blue', label='Overall')
    uber_label = patches.Patch(color='green', label='Uber')
    lyft_label = patches.Patch(color='red', label='Lyft')
    plt.legend(handles=[overall_label, uber_label, lyft_label], loc='upper right')
    plt.show()

    print('END AVERAGE PRICE DISTANCE RATIO BY HOUR')
    print('________________________________________________________________\n')

# analyses surge multiplier by hour (only lyft has surge mult)
def average_surge_mult_by_hour(lyft_frame):
    print('________________________________________________________________\n')
    print('AVERAGE SURGE MULTIPLIER BY HOUR:')

    # finding average surge multiplier for lyft
    lyft_avg_surge_df = lyft_frame.groupby(['hour'])['surge_multiplier'].mean()

    # printing results
    print('Lyft:')
    print(lyft_avg_surge_df)
    
    # plotting line chart
    plt.figure()
    lyft_avg_surge_df.plot(kind='line', color='red')
    plt.title('Average Surge Multiplier by Hour (Lyft)')
    plt.xlabel('Hour')
    plt.ylabel('Average Surge Multiplier')
    plt.show()

    print('END AVERAGE SURGE MULTIPLIER BY HOUR')
    print('________________________________________________________________\n')

# showing average rent by neighborhood (this is just the cleaned rent df)
def average_rent_by_neighborhood(rent_df):
    print('________________________________________________________________\n')
    print('AVERAGE RENT BY NEIGHBORHOOD:')

    print(rent_df)
    plt.figure()
    rent_df.plot(kind='bar')
    plt.title('Average Rent by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Average Rent (USD)')
    plt.legend('', frameon=False)
    plt.show()

    print('END AVERAGE RENT BY NEIGHBORHOOD')
    print('________________________________________________________________\n')

# attempting to make a map with the coordinates of rides... they weren't spread enough
def price_map(rideshare_df):    
    print('________________________________________________________________\n')
    print('PRICE MAP')

    # the coordinates are just one per neighborhood, which is why this map doesn't work
    print(rideshare_df[['latitude', 'longitude']].value_counts())
    plt.figure()
    plt.scatter(rideshare_df['longitude'], rideshare_df['latitude'], s=1, c=rideshare_df['price'], alpha=0.5)
    plt.title('(Attempted) Price Map of Boston')
    plt.xlabel('Longitude')
    plt.xlabel('Latitude')
    plt.show()

    print('END PRICE MAP')
    print('________________________________________________________________\n')

# average price/rent by neighborhood
def average_price_rent_by_neighborhood(rideshare_df, rent_df):
    print('________________________________________________________________\n')
    print('AVERAGE PRICE AND RENT BY NEIGHBORHOOD')
    
    # source average price frame
    source_price_df = rideshare_df.groupby(['source'])['price'].mean().to_frame()
    source_price_df.columns = ['avg_source_price']
    
    # destination average price frame
    destination_price_df = rideshare_df.groupby(['destination'])['price'].mean().to_frame()
    destination_price_df.columns = ['avg_destination_price']
    
    # adding in rent price in thousands
    rent_price_df = rent_df
    rent_price_df = rent_price_df.reset_index()
    rent_price_df.columns = ['source', 'avg_rent_thousands']
    rent_price_df = rent_price_df.set_index(['source'])
    
    # joining on index and dividing rent into thousands
    neighborhood_df = source_price_df.merge(destination_price_df, how='left', left_index=True, right_index=True)
    neighborhood_df = neighborhood_df.merge(rent_price_df, how='left', left_index=True, right_index=True)
    neighborhood_df['avg_rent_thousands'] = neighborhood_df['avg_rent_thousands'] / 1000

    # modest inverse correlation
    print(neighborhood_df)
    print(neighborhood_df.corr().to_string())
    
    # weak relationship between rent and rideshare prices
    plt.figure()
    neighborhood_df.plot(kind='bar')
    plt.title('Average Pricing and Rent by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.xticks(fontsize=8)
    plt.ylabel('Average Price (USD) or Rent (thousands USD)')
    source_label = patches.Patch(color='#1f77b4', label='Avg. Source Price (USD)')
    dest_label = patches.Patch(color='#ff7f0e', label='Avg. Dest. Price (USD)')
    rent_label = patches.Patch(color='#2ca02c', label='Avg. Rent (thousands USD)')
    plt.legend(handles=[source_label, dest_label, rent_label], loc='upper right')
    plt.show()
    
    print('END AVERAGE PRICE AND RENT BY NEIGHBORHOOD')
    print('________________________________________________________________\n')

"""
    6. CORRELATION MATRICES
"""

# creates overall correlation matrix
def overall_correlation_matrix(rideshare_df):
    print('________________________________________________________________\n')
    print('OVERALL CORRELATION MATRIX')
    
    # getting correlation matrix of variables to price
    price_correlation_df = rideshare_df.corr()['price'].to_frame()
    price_correlation_df.columns = ['correlation']
    
    # recording direction then removing sign from correlation
    price_correlation_df['relationship'] = price_correlation_df['correlation'].apply(lambda x: 'direct' if x >= 0 else 'inverse')
    price_correlation_df['correlation'] = price_correlation_df['correlation'].apply(lambda x: abs(x))
    
    # sorting descending by correlation and removing price
    price_correlation_df = price_correlation_df.sort_values(['correlation'], ascending=False)
    price_correlation_df = price_correlation_df[1:]
    
    # pruning out price_dist_ratio because it can't be used for regression
    price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)
    
    # really only the top 5 or so matter, the rest drop off quickly
    price_correlation_df = price_correlation_df[0:5]
    
    price_correlation_df = price_correlation_df.reset_index()
    price_correlation_df.columns = ['variable', 'correlation', 'relationship']
    
    # printing the result and plotting
    print(price_correlation_df)
    
    plt.figure()
    plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'blue','inverse':'black'}))
    plt.xticks(rotation=45)
    plt.xlabel('Variable')
    plt.ylabel('Correlation Strength')
    plt.title('Overall Variable Correlations (Top 5)')
    direct_label = patches.Patch(color='blue', label='Direct')
    inverse_label = patches.Patch(color='black', label='Inverse')
    plt.legend(handles=[direct_label, inverse_label], loc='upper right')
    plt.show()

    print('END OVERALL CORRELATION MATRIX')
    print('________________________________________________________________\n')
    
# creates correlation matrix for uber
def uber_correlation_matrix(uber_frame):
    print('________________________________________________________________\n')
    print('UBER CORRELATION MATRIX')
    
    # getting correlation matrix of variables to price
    price_correlation_df = uber_frame.corr()['price'].to_frame()
    price_correlation_df.columns = ['correlation']
    
    # recording direction then removing sign from correlation
    price_correlation_df['relationship'] = price_correlation_df['correlation'].apply(lambda x: 'direct' if x >= 0 else 'inverse')
    price_correlation_df['correlation'] = price_correlation_df['correlation'].apply(lambda x: abs(x))
    
    # sorting descending by correlation and removing price
    price_correlation_df = price_correlation_df.sort_values(['correlation'], ascending=False)
    price_correlation_df = price_correlation_df[1:]
    
    # pruning out price_dist_ratio because it can't be used for regression
    price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)
    
    # really only the top 5 or so matter, the rest drop off quickly
    price_correlation_df = price_correlation_df[0:5]
    
    price_correlation_df = price_correlation_df.reset_index()
    price_correlation_df.columns = ['variable', 'correlation', 'relationship']
    
    # printing the result and plotting
    print(price_correlation_df)
    
    plt.figure()
    plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'green','inverse':'black'}))
    plt.xticks(rotation=45)
    plt.xlabel('Variable')
    plt.ylabel('Correlation Strength')
    plt.title('Uber Variable Correlations (Top 5)')
    direct_label = patches.Patch(color='green', label='Direct')
    inverse_label = patches.Patch(color='black', label='Inverse')
    plt.legend(handles=[direct_label, inverse_label], loc='upper right')
    plt.show()

    print('END UBER CORRELATION MATRIX')
    print('________________________________________________________________\n')

# creates correlation matrix for lyft
def lyft_correlation_matrix(lyft_frame):
    print('________________________________________________________________\n')
    print('LYFT CORRELATION MATRIX')
    
    # getting correlation matrix of variables to price
    price_correlation_df = lyft_frame.corr()['price'].to_frame()
    price_correlation_df.columns = ['correlation']
    
    # recording direction then removing sign from correlation
    price_correlation_df['relationship'] = price_correlation_df['correlation'].apply(lambda x: 'direct' if x >= 0 else 'inverse')
    price_correlation_df['correlation'] = price_correlation_df['correlation'].apply(lambda x: abs(x))
    
    # sorting descending by correlation and removing price
    price_correlation_df = price_correlation_df.sort_values(['correlation'], ascending=False)
    price_correlation_df = price_correlation_df[1:]
    
    # pruning out price_dist_ratio because it can't be used for regression
    price_correlation_df = price_correlation_df.drop(['price_dist_ratio'], axis=0)
    
    # really only the top 5 or so matter, the rest drop off quickly
    price_correlation_df = price_correlation_df[0:5]
    
    price_correlation_df = price_correlation_df.reset_index()
    price_correlation_df.columns = ['variable', 'correlation', 'relationship']
    
    # printing the result and plotting
    print(price_correlation_df)
    
    plt.figure()
    plt.bar(x=price_correlation_df['variable'], height=price_correlation_df['correlation'], color=price_correlation_df['relationship'].map({'direct':'red','inverse':'black'}))
    plt.xticks(rotation=45)
    plt.xlabel('Variable')
    plt.ylabel('Correlation Strength')
    plt.title('Lyft Variable Correlations (Top 5)')
    direct_label = patches.Patch(color='red', label='Direct')
    inverse_label = patches.Patch(color='black', label='Inverse')
    plt.legend(handles=[direct_label, inverse_label], loc='upper right')
    plt.show()

    print('END LYFT CORRELATION MATRIX')
    print('________________________________________________________________\n')

"""
    7. LINEAR REGRESSION
"""

# creates a few overall linear regression models
def overall_linear_regression(rideshare_df):
    print('________________________________________________________________\n')
    print('OVERALL LINEAR REGRESSION:')

    # starting with just distance
    print('\nDistance Model:')
    distance_model = ols("price ~ distance", rideshare_df).fit()
    print(distance_model.summary())    
    
    # trying just surge
    print('\nSurge Model:')
    surge_model = ols("price ~ surge_multiplier", rideshare_df).fit()
    print(surge_model.summary())
    
    # trying just cab type
    print('\nCab Type Model:')    
    cab_type_model = ols("price ~ is_uber", rideshare_df).fit()
    print(cab_type_model.summary())
    
    # trying distance, surge, cab type, and the rents (top 5, these were the only variables with notable correlations)
    print('\nCombo Model:')
    combo_model = ols("price ~ distance + surge_multiplier + is_uber + source_rent + destination_rent", rideshare_df).fit()
    print(combo_model.summary())
    
    # plotting scatterplot for the combo model
    rideshare_df['predicted_price'] = combo_model.predict(rideshare_df[['distance', 'surge_multiplier', 'is_uber', 'source_rent', 'destination_rent']])
    
    plt.figure()
    # actual prices in black, predicted in blue
    plt.scatter(rideshare_df['distance'], rideshare_df['price'], s=0.1, c='black')
    plt.scatter(rideshare_df['distance'], rideshare_df['predicted_price'], s=0.1, c='blue')
    plt.title('Price vs. Distance (Overall)')
    plt.xlabel('Distance (Miles)')
    plt.ylabel('Price (USD)')
    predicted_label = patches.Patch(color='blue', label='Predicted')
    price_label = patches.Patch(color='black', label='Actual')
    plt.legend(handles=[predicted_label, price_label], loc='upper right')
    plt.show()
    
    print('END OVERALL LINEAR REGRESSION')
    print('________________________________________________________________\n')

# creates ols model for uber
def uber_linear_regression(uber_frame):
    print('________________________________________________________________\n')
    print('OVERALL UBER REGRESSION:')

    # starting with just distance
    print('\nDistance Model:')
    distance_model = ols("price ~ distance", uber_frame).fit()
    print(distance_model.summary())    
    
    # trying distance and the rents
    print('\nCombo Model:')
    combo_model = ols("price ~ distance + source_rent + destination_rent", uber_frame).fit()
    print(combo_model.summary())
    
    # plotting scatterplot for the combo model
    uber_frame['predicted_price'] = combo_model.predict(uber_frame[['distance', 'source_rent', 'destination_rent']])
    
    plt.figure()
    # actual prices in black, predicted in green
    plt.scatter(uber_frame['distance'], uber_frame['price'], s=0.1, c='black')
    plt.scatter(uber_frame['distance'], uber_frame['predicted_price'], s=0.1, c='green')
    plt.title('Price vs. Distance (Uber)')
    plt.xlabel('Distance (Miles)')
    plt.ylabel('Price (USD)')
    predicted_label = patches.Patch(color='green', label='Predicted')
    price_label = patches.Patch(color='black', label='Actual')
    plt.legend(handles=[predicted_label, price_label], loc='upper right')
    plt.show()
    
    print('END OVERALL UBER REGRESSION')
    print('________________________________________________________________\n')

# creates ols models for lyft
def lyft_linear_regression(lyft_frame):
    print('________________________________________________________________\n')
    print('LYFT LINEAR REGRESSION:')

    # starting with just distance
    print('\nDistance Model:')
    distance_model = ols("price ~ distance", lyft_frame).fit()
    print(distance_model.summary())    
    
    # trying just surge
    print('\nSurge Model:')
    surge_model = ols("price ~ surge_multiplier", lyft_frame).fit()
    print(surge_model.summary())
        
    # trying distance, surge, and the rents
    print('\nCombo Model:')
    combo_model = ols("price ~ distance + surge_multiplier + source_rent + destination_rent", lyft_frame).fit()
    print(combo_model.summary())
    
    # plotting scatterplot for the combo model
    lyft_frame['predicted_price'] = combo_model.predict(lyft_frame[['distance', 'surge_multiplier', 'source_rent', 'destination_rent']])
    
    plt.figure()
    # actual prices in black, predicted in red
    plt.scatter(lyft_frame['distance'], lyft_frame['price'], s=0.1, c='black')
    plt.scatter(lyft_frame['distance'], lyft_frame['predicted_price'], s=0.1, c='red')
    plt.title('Price vs. Distance (Lyft)')
    plt.xlabel('Distance (Miles)')
    plt.ylabel('Price (USD)')
    predicted_label = patches.Patch(color='red', label='Predicted')
    price_label = patches.Patch(color='black', label='Actual')
    plt.legend(handles=[predicted_label, price_label], loc='upper right')
    plt.show()
    
    print('END LYFT LINEAR REGRESSION')
    print('________________________________________________________________\n')

"""
    8. MAIN FUNCTION
"""

# overarching program logic; this will run straight through, leave a record on the console, and create plots
def main():
    # start notice
    print('Initializing...')
    
    # loading data from csvs
    rideshare_df = load_rideshare()
    rent_df = load_rents()
    
    # cleaning data
    rideshare_df = clean_rideshare(rideshare_df)
    rent_df = clean_rent(rent_df, rideshare_df)
    
    # adding features
    rideshare_df = add_is_uber(rideshare_df)
    rideshare_df = add_price_dist_ratio(rideshare_df)
    rideshare_df = add_average_rents(rideshare_df, rent_df)
    uber_frame = get_uber_frame(rideshare_df)
    lyft_frame = get_lyft_frame(rideshare_df)
    
    # distribution exploration
    prelim_exploration(rideshare_df)
    value_count_exploration(rideshare_df)
    name_exploration(rideshare_df)

    # graphical analyses
    average_price_by_name(uber_frame, lyft_frame)
    total_price_by_name(uber_frame, lyft_frame)
    average_price_by_weather(rideshare_df, uber_frame, lyft_frame)
    average_distance_by_weather(rideshare_df, uber_frame, lyft_frame)
    average_price_by_hour(rideshare_df, uber_frame, lyft_frame)
    average_price_dist_ratio_by_hour(rideshare_df, uber_frame, lyft_frame)
    average_surge_mult_by_hour(lyft_frame)
    average_rent_by_neighborhood(rent_df)
    price_map(rideshare_df)
    average_price_rent_by_neighborhood(rideshare_df, rent_df)
    
    # correlation
    overall_correlation_matrix(rideshare_df)
    uber_correlation_matrix(uber_frame)
    lyft_correlation_matrix(lyft_frame)
    
    # linear regression
    overall_linear_regression(rideshare_df)
    uber_linear_regression(uber_frame)
    lyft_linear_regression(lyft_frame)
    
    # exit notice
    print('Exiting...')

"""
    9. MAIN CALL
"""

main()
