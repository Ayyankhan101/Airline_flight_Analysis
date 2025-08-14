import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Airline Flight Analysis Dashboard", layout="wide")

st.title('Airline Flight Analysis Dashboard')

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('airlines_flights_data.csv')
    df.drop_duplicates(inplace=True)
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    df['route'] = df['source_city'] + ' - ' + df['destination_city']
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header('Filters for EDA')
airline = st.sidebar.multiselect(
    'Select Airline',
    options=df['airline'].unique(),
    default=df['airline'].unique()
)

source_city = st.sidebar.multiselect(
    'Select Source City',
    options=df['source_city'].unique(),
    default=df['source_city'].unique()
)

destination_city = st.sidebar.multiselect(
    'Select Destination City',
    options=df['destination_city'].unique(),
    default=df['destination_city'].unique()
)

flight_class = st.sidebar.multiselect(
    'Select Class',
    options=df['class'].unique(),
    default=df['class'].unique()
)

# Filter the dataframe
df_filtered = df[df['airline'].isin(airline) & df['source_city'].isin(source_city) & df['destination_city'].isin(destination_city) & df['class'].isin(flight_class)]

# Create tabs
tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Price Prediction"])

with tab1:
    st.header('Exploratory Data Analysis')

    st.header('Filtered Data')
    st.write(df_filtered)

    # Visualizations
    st.header('Data Visualizations')

    with st.expander("Price Distribution"):
        st.markdown("#### Distribution of Flight Prices")
        fig, ax = plt.subplots()
        sns.histplot(df_filtered['price'], bins=30, kde=True, ax=ax, color='green' , alpha=0.6)
        st.pyplot(fig)
        st.markdown("The distribution of flight prices is right-skewed, indicating that most flights have prices on the lower end, with a few flights having very high prices.")

    with st.expander("Flight Counts"):
        st.markdown("#### Number of Flights by Airline")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(y='airline', data=df_filtered, order = df_filtered['airline'].value_counts().index, ax=ax , palette='viridis')
        st.pyplot(fig)
        st.markdown("This plot shows the number of flights operated by each airline in the dataset. Vistara and Air India have the highest number of flights.")

        st.markdown("#### Number of Flights by Class")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='class', data=df_filtered, palette='pastel', ax=ax)
        st.pyplot(fig)
        st.markdown("This plot shows that the dataset contains significantly more Economy class flights than Business class flights.")


    with st.expander("Price Analysis"):
        st.markdown("#### Price Distribution by Airline")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='airline', y='price', data=df_filtered, ax=ax , palette='Set2')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("This box plot shows the distribution of prices for each airline. Vistara and Air India have a wider range of prices, which is consistent with them operating more flights. We can also see that some airlines have higher median prices than others.")

        st.markdown("#### Price Distribution by Class")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='class', y='price', data=df_filtered, palette='Set1', ax=ax)
        st.pyplot(fig)
        st.markdown("As expected, Business class flights are significantly more expensive than Economy class flights.")

        st.markdown("#### Price vs. Number of Stops")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='stops', y='price', data=df_filtered, ax=ax, palette='coolwarm')
        st.pyplot(fig)
        st.markdown("This box plot shows the relationship between the number of stops and the price. Flights with one stop seem to have the highest median price.")

    with st.expander("Route Analysis"):
        st.markdown("#### Most Popular Routes")
        fig, ax = plt.subplots(figsize=(12, 8))
        df_filtered['route'].value_counts().nlargest(10).plot(kind='barh', color='skyblue', ax=ax)
        plt.title('Top 10 Most Popular Routes')
        plt.xlabel('Number of Flights')
        plt.ylabel('Route')
        st.pyplot(fig)

        st.markdown("#### Average Price by Route")
        fig, ax = plt.subplots(figsize=(12, 8))
        df_filtered.groupby('route')['price'].mean().nlargest(10).sort_values(ascending=True).plot(kind='barh', color='coral', ax=ax)
        plt.title('Top 10 Most Expensive Routes by Average Price')
        plt.xlabel('Average Price')
        plt.ylabel('Route')
        st.pyplot(fig)

with tab2:
    st.header("Price Prediction")

    st.markdown("""
    This section allows you to predict the price of a flight based on different features. 
    Please select the features for the flight you want to predict.
    """)

    # Train the model
    @st.cache_resource
    def train_model():
        # Select features and target
        features = ['airline', 'source_city', 'destination_city', 'stops', 'class', 'duration', 'days_left']
        target = 'price'

        X = df[features]
        y = df[target]

        # Identify categorical and numerical features
        categorical_features = ['airline', 'source_city', 'destination_city', 'stops', 'class']
        numerical_features = ['duration', 'days_left']

        # Create a preprocessor for categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Create a pipeline with the preprocessor and the model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

        # Train the model
        model.fit(X, y)
        return model

    model = train_model()

    # User input for prediction
    pred_airline = st.selectbox('Airline', options=df['airline'].unique())
    pred_source_city = st.selectbox('Source City', options=df['source_city'].unique())
    pred_destination_city = st.selectbox('Destination City', options=df['destination_city'].unique())
    pred_stops = st.selectbox('Stops', options=df['stops'].unique())
    pred_class = st.selectbox('Class', options=df['class'].unique())
    pred_duration = st.number_input('Duration (hours)', min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    pred_days_left = st.number_input('Days Left to Departure', min_value=1, max_value=50, value=15, step=1)

    # Create a dataframe from the user input
    input_data = pd.DataFrame({
        'airline': [pred_airline],
        'source_city': [pred_source_city],
        'destination_city': [pred_destination_city],
        'stops': [pred_stops],
        'class': [pred_class],
        'duration': [pred_duration],
        'days_left': [pred_days_left]
    })

    # Predict the price
    if st.button('Predict Price'):
        prediction = model.predict(input_data)
        st.success(f'The predicted price of the flight is: ₹ {prediction[0]:.2f}')
