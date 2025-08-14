# Airline Flight Price Analysis and Prediction

This project provides a comprehensive analysis of airline flight data, uncovering insights into pricing, airline performance, and travel patterns. It features an interactive dashboard built with Streamlit that allows users to explore the data and predict flight prices using a machine learning model.

## Key Features

- **Exploratory Data Analysis (EDA):** In-depth analysis of flight data to understand the relationships between different variables.
- **Interactive Dashboard:** A user-friendly Streamlit dashboard to visualize the data and interact with the model.
- **Price Prediction:** A machine learning model that predicts flight prices based on various features.
- **Feature Importance:** Analysis of which factors are most influential in determining flight prices.

## Technologies Used

- **Python:** The core programming language for the project.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For building the machine learning model.
- **Streamlit:** For creating the interactive web dashboard.

## Dataset

The dataset used in this project contains information about airline flights, including:

- **Airline:** The name of the airline.
- **Source & Destination:** The departure and arrival cities.
- **Class:** The travel class (Economy or Business).
- **Duration:** The flight duration in hours.
- **Stops:** The number of stops.
- **Days Left:** The number of days before departure.
- **Price:** The price of the flight ticket.

## Analysis and Visualizations

The project includes a detailed exploratory data analysis with various visualizations to highlight key findings:

- **Price Distribution:** The distribution of flight prices is right-skewed, with most tickets in the lower price range.
- **Airline Performance:** Vistara and Air India have the highest number of flights and a wider range of prices.
- **Price by Class:** Business class tickets are significantly more expensive than Economy class tickets.
- **Price vs. Duration:** There is a positive correlation between flight duration and price.
- **Price vs. Days Left:** Flight prices tend to increase as the departure date approaches.

## Predictive Modeling

A Random Forest Regressor model was trained to predict flight prices with an R-squared score of **0.98**, indicating a high level of accuracy. The most important features for predicting the price are:

1.  **Class:** Whether the flight is Economy or Business.
2.  **Duration:** The length of the flight.
3.  **Days Left:** The number of days until departure.

## How to Run the Dashboard

To run the interactive dashboard on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/airline-flight-analysis.git
    cd airline-flight-analysis
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run dashboard.py
    ```

This will open the dashboard in your web browser, where you can explore the data and make price predictions.