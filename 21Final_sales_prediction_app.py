import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go 


st.image("/Users/michaellos1/Downloads/7451e6406611ffc22724a4881f47f9222a6d1315.png", width=100)


with st.sidebar:
    selected_option = st.radio(
        "Navigation",
        [
            "🤖 Forecast",
            "📤 Upload CSV",
            "📊 Data Insights",
            "💬 Chatbot",
            "🚪 Exit"
        ]
    )


feature_explanations = {
    "Forecast": "This feature allows users to input sales-related data and predicts weekly sales using a trained machine learning model. "
                "It leverages engineered features to capture seasonality, trends, and store-specific behaviors. "
                "Predictions help users plan inventory, promotions, and staffing effectively.",
    
    "Upload CSV": "This feature allows users to upload their own CSV files with relevant sales data. "
                  "The uploaded data is preprocessed using the same pipeline as the training data to ensure consistency. "
                  "Predictions are generated for each record in the file, providing batch insights.",
    
    "Data Insights": "This section enables users to explore the training data used for building the model. "
                     "Users can view summary statistics, visualize feature correlations, and analyze data distributions. "
                     "It provides insights into data quality, trends, and relationships between features.",
    
    "Model Evaluation": "Evaluates the performance of the trained model on metrics such as RMSE, MAE, and R². "
                        "Visualizations like actual vs predicted plots help users understand the accuracy of predictions. "
                        "This section ensures the model is reliable for making business decisions.",
    
    "Chatbot": "Provides users with a Q&A interface to clarify any questions about the features and navigation of the app. "
               "Users can ask about specific features, their importance, or how they are used in predictions. "
               "It serves as an interactive guide for better user experience.",
    
    "Exit": "Allows users to exit the application with a thank-you message and farewell animation. "
            "This feature provides a smooth transition when the app session ends. "
            "It ensures a polished and user-friendly experience.",
    
    "Rolling 7-day average sales": "The average weekly sales over the last 7 days, smoothing out short-term fluctuations. "
                                   "It helps capture recent trends without overreacting to daily variability. "
                                   "This feature is critical for understanding near-term sales patterns.",
    
    "Lag_1_Week": "The sales value from one week ago for the same store and department. "
                  "It captures short-term trends and acts as a key indicator for weekly seasonality. "
                  "This feature is vital for modeling the effect of last week's performance on the current week.",
    
    "Lag_2_Weeks": "The sales value from two weeks ago for the same store and department. "
                   "This feature extends the trend-capturing window to medium-term variations in sales. "
                   "It complements other lag features for a holistic view of past performance.",
    
    "Lag_3_Weeks": "The sales value from three weeks ago for the same store and department. "
                   "This feature tracks longer-term sales trends and seasonality. "
                   "It aids in identifying recurring patterns over a month-long period.",
    
    "Lag_4_Weeks": "The sales value from four weeks ago for the same store and department. "
                   "This feature captures monthly seasonality and recurring cycles in sales. "
                   "It is particularly useful for holidays and promotions that occur monthly.",
    
    "Dept_Avg_Sales": "The average sales for a department across all stores. "
                      "This feature provides insights into department-wide performance and benchmarks. "
                      "It helps identify underperforming or outperforming departments.",
    
    "Store_Avg_Sales": "The average sales for a store across all departments. "
                       "This feature gives an overview of store-specific performance trends. "
                       "It is essential for understanding the store's overall sales health.",
    
    "Holiday_Impact": "Indicates the effect of holidays on sales by multiplying a holiday indicator with weekly sales. "
                      "It captures spikes or drops in sales due to holidays, promotions, or special events. "
                      "This feature is crucial for understanding and modeling seasonal variations.",
    
    "Sales_Per_Sqft": "Weekly sales divided by the store size, normalizing sales relative to the store area. "
                      "This feature allows for better comparison of sales performance across stores of varying sizes. "
                      "It provides a measure of efficiency in utilizing store space for generating revenue.",
    
    "Week_Sin": "The sine of the week number, representing cyclical weekly patterns in sales. "
                "It helps the model capture seasonal variations across the 52 weeks of the year. "
                "This feature is paired with Week_Cos to fully represent the cyclical nature of weeks.",
    
    "Week_Cos": "The cosine of the week number, complementing Week_Sin to capture cyclical weekly patterns. "
                "Together with Week_Sin, it ensures that the model understands seasonal behaviors effectively. "
                "This feature is particularly useful for detecting peaks and troughs in sales trends.",
    
    "Rolling_30_day_avg": "The average weekly sales over the last 30 days, smoothing out medium-term fluctuations. "
                          "It highlights broader trends while ignoring short-term noise. "
                          "This feature is helpful for understanding sustained sales performance.",
    
    "MarkDown1": "Represents discounts or promotions applied to products, affecting sales during specific periods. "
                 "It captures the impact of promotional strategies on consumer behavior. "
                 "This feature is particularly relevant for marketing-driven sales spikes.",
    
    "MarkDown2": "Discounts or promotions applied to products, focusing on a specific category or timeframe. "
                 "It helps model the effect of seasonal or event-specific markdowns. "
                 "This feature complements other markdown features for a complete view of promotional impacts.",
    
    "MarkDown3": "Another promotional feature capturing specific types of discounts or offers. "
                 "It reflects the targeted marketing efforts for certain products or departments. "
                 "Including this feature helps model the effect of multi-layered promotions.",
    
    "MarkDown4": "Captures the impact of a unique category of markdowns applied to products. "
                 "It focuses on periodic or region-specific discount strategies. "
                 "This feature supports a more detailed understanding of sales dynamics.",
    
    "MarkDown5": "Represents additional discounts or promotions applied to products. "
                 "This feature tracks seasonal or end-of-season clearance markdowns. "
                 "It is key to understanding sales surges during clearance periods.",
    
    "Correlation Heatmap": "Visualizes correlations between numeric features to identify relationships and trends. "
                           "Strong correlations highlight key predictors of sales performance. "
                           "This analysis is critical for feature selection and understanding data dependencies.",
    
    "Temperature": "Represents the average temperature during the sales week. "
                   "It captures the effect of weather on consumer behavior and purchasing patterns. "
                   "This feature is particularly relevant for seasonal goods or regions with varying climates.",
    
    "Fuel_Price": "The average price of fuel during the sales week. "
                  "It indicates external economic factors that might influence spending behavior. "
                  "This feature is especially useful for modeling sales in regions dependent on automotive travel.",
    
    "Summary Statistics": "This section provides descriptive statistics for numeric features in the dataset. "
                          "Metrics such as mean, median, standard deviation, and quartiles are calculated to summarize the data. "
                          "It helps identify trends, outliers, and data quality issues.",
    
    "Feature Distributions": "This section visualizes the distribution of individual features using histograms or density plots. "
                             "Users can explore how values are spread across the range and identify skewness or clustering. "
                             "This is useful for understanding data variability and informing preprocessing decisions.",
    
    "Average Weekly Sales per Week (2010-2012)": "This visualization shows trends in average weekly sales over the years 2010, 2011, and 2012. "
                                                 "It highlights seasonal patterns, yearly differences, and significant peaks or drops. "
                                                 "This insight is critical for planning promotions and forecasting future sales."
}


@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = pickle.load(open("Final_model.sav", "rb"))
        preprocessor = pickle.load(open("Final_preprocessor_pipeline.sav", "rb"))
    except FileNotFoundError:
        st.error("Model files not found. Ensure the model and preprocessor are in the working directory.")
        st.stop()
    return model, preprocessor


@st.cache_resource
def load_and_merge_data():
    try:
        
        features_url = "https://drive.google.com/file/d/1J80aXBMcYp54F759o8wg5Ud0Lo_FtK62/view?usp=sharing"
        stores_url = "https://drive.google.com/file/d/1ZoGRulYebRMdjBaVsTz9KkqbngnNxn9b/view?usp=sharing"
        train_url = "https://drive.google.com/file/d/1H82PVxZHJJUOiUShGWZzXK_VoB8-XKy7/view?usp=sharing"

        def get_direct_download_url(drive_url):
            return 'https://drive.google.com/uc?export=download&id=' + drive_url.split('/')[-2]

        features = pd.read_csv(get_direct_download_url(features_url))
        stores = pd.read_csv(get_direct_download_url(stores_url))
        train = pd.read_csv(get_direct_download_url(train_url))

        features['CPI'] = features['CPI'].fillna(features['CPI'].mean())
        features['Unemployment'] = features['Unemployment'].fillna(features['Unemployment'].mean())
        markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        features[markdown_columns] = features[markdown_columns].fillna(0)

        # Merge datasets
        train_features = pd.merge(train, features, on=['Store', 'Date'], how='left')
        train_merged = pd.merge(train_features, stores, on='Store', how='left')
        train_merged.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

        # Feature engineering
        train_merged['Date'] = pd.to_datetime(train_merged['Date'])
        train_merged['Year'] = train_merged['Date'].dt.year
        train_merged['Month'] = train_merged['Date'].dt.month
        train_merged['Week'] = train_merged['Date'].dt.isocalendar().week
        train_merged['DayOfWeek'] = train_merged['Date'].dt.dayofweek
        train_merged['Holiday_Impact'] = train_merged['IsHoliday'] * train_merged['Weekly_Sales']
        train_merged['Store_Avg_Sales'] = train_merged.groupby('Store')['Weekly_Sales'].transform('mean')
        train_merged['Dept_Avg_Sales'] = train_merged.groupby('Dept')['Weekly_Sales'].transform('mean')
        train_merged['Lag_1_Week'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).fillna(0)
        train_merged['Lag_2_Weeks'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2).fillna(0)
        train_merged['Lag_3_Weeks'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(3).fillna(0)
        train_merged['Lag_4_Weeks'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4).fillna(0)
        train_merged['Rolling_7_day_avg'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
        train_merged['Rolling_30_day_avg'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()).fillna(0)
        train_merged['Week_Sin'] = np.sin(2 * np.pi * train_merged['Week'] / 52)
        train_merged['Week_Cos'] = np.cos(2 * np.pi * train_merged['Week'] / 52)
        train_merged['Sales_Per_Sqft'] = train_merged['Weekly_Sales'] / train_merged['Size']

        return train_merged

    except Exception as e:
        st.error(f"An error occurred while loading or merging the datasets: {e}")
        st.stop()

model, preprocessor = load_model_and_preprocessor()
train_merged = load_and_merge_data()

st.title("🤖 Weekly Sales Forecast and Analysis")

if selected_option == "🤖 Forecast":
    st.subheader("📈 Weekly Sales Prediction")
    st.write("Enter the input features to predict weekly sales.")

    inputs = {
        "Rolling_7_day_avg": st.number_input("Rolling 7-day average sales", min_value=0.0, step=100.0, value=21682.14),
        "Lag_1_Week": st.number_input("Sales from 1 week ago", min_value=0.0, step=100.0, value=21810.36),
        "Lag_2_Weeks": st.number_input("Sales from 2 weeks ago", min_value=0.0, step=100.0, value=21611.33),
        "Lag_3_Weeks": st.number_input("Sales from 3 weeks ago", min_value=0.0, step=100.0, value=23146.95),
        "Dept_Avg_Sales": st.number_input("Department average sales", min_value=0.0, step=100.0, value=30663.80),
        "Store_Avg_Sales": st.number_input("Store average sales", min_value=0.0, step=100.0, value=13546.21),
        "Size": st.number_input("Store size (in sqft)", min_value=0.0, step=500.0, value=158114.0),
        "Holiday_Impact": st.number_input("Holiday impact", min_value=0.0, step=1.0, value=0.0),
        "MarkDown5": st.number_input("Markdown 5", min_value=0.0, step=10.0, value=0.0),
        "Week_Sin": st.number_input("Sine of week", min_value=-1.0, max_value=1.0, step=0.1, value=0.57),
        "Week_Cos": st.number_input("Cosine of week", min_value=-1.0, max_value=1.0, step=0.1, value=-0.82),
        "Rolling_30_day_avg": st.number_input("Rolling 30-day average sales", min_value=0.0, step=100.0, value=22990.73),
        "Sales_Per_Sqft": st.number_input("Sales per square foot", min_value=0.0, step=1.0, value=0.13),
        "Lag_4_Weeks": st.number_input("Sales from 4 weeks ago", min_value=0.0, step=100.0, value=20339.67),
    }

    if st.button("Predict Weekly Sales"):
        new_data = pd.DataFrame([inputs])
        try:
            transformed_data = preprocessor.transform(new_data)
            prediction = model.predict(transformed_data)
            margin_of_error = prediction[0] * 0.0184
            st.success(f"Predicted Weekly Sales: ${prediction[0]:,.2f} ± ${margin_of_error:,.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")

elif selected_option == "📊 Data Insights":
    st.subheader("📊 Data Insights")
    st.write("Gain insights into the training data and explore important features.")

    input_features = [
        "Rolling_7_day_avg", "Lag_1_Week", "Lag_2_Weeks", "Lag_3_Weeks",
        "Dept_Avg_Sales", "Store_Avg_Sales", "Size", "Holiday_Impact",
        "MarkDown5", "Week_Sin", "Week_Cos", "Rolling_30_day_avg",
        "Sales_Per_Sqft", "Lag_4_Weeks"
    ]

    columns_to_check = [
        "Rolling_7_day_avg", "Lag_1_Week", "Lag_2_Weeks", "Lag_3_Weeks",
        "Dept_Avg_Sales", "Store_Avg_Sales", "Size", "MarkDown5",
        "Rolling_30_day_avg", "Sales_Per_Sqft", "Lag_4_Weeks"
    ]
    for col in columns_to_check:
        train_merged.loc[train_merged[col] < 0, col] = np.nan

    st.write("### Cleaned Summary Statistics")
    st.write(train_merged[input_features].describe())

    st.write("### Correlation Heatmap")
    numeric_columns = train_merged[input_features].select_dtypes(include=np.number)

    corr_matrix = numeric_columns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    st.pyplot(fig)

    st.write("### Feature Distributions")
    selected_feature = st.selectbox("Select a Feature to Plot", input_features)

    if selected_feature in numeric_columns:
        fig, ax = plt.subplots()
        sns.histplot(train_merged[selected_feature], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The selected feature is not numeric and cannot be plotted.")

    import plotly.io as pio
    pio.templates["custom_dark"] = pio.templates["plotly_dark"]
    pio.templates["custom_dark"].layout.paper_bgcolor = "black"
    pio.templates["custom_dark"].layout.plot_bgcolor = "black"
    pio.templates.default = "custom_dark"

    st.write("### Average Weekly Sales: Holidays vs Non-Holidays")
    holiday_sales = train_merged.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

    import plotly.express as px
    fig = px.bar(
        holiday_sales,
        x='IsHoliday',
        y='Weekly_Sales',
        title='Average Weekly Sales: Holidays vs Non-Holidays',
        labels={'IsHoliday': 'Is Holiday', 'Weekly_Sales': 'Average Weekly Sales ($)'},
        color='IsHoliday',
        color_discrete_map={0: 'blue', 1: 'magenta'}
    )
    fig.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['Non-Holiday', 'Holiday'])
    st.plotly_chart(fig)

if selected_option == "📊 Data Insights":
    st.subheader("📊 Data Insights")
    st.write("Gain insights into the training data and explore important features.")

    st.write("### Weekly Sales During Holidays vs Non-Holidays")
    holi_wk_sales_no = train_merged[train_merged['IsHoliday'] == 0].groupby('Week')['Weekly_Sales'].mean()
    holi_wk_sales_yes = train_merged[train_merged['IsHoliday'] == 1].groupby('Week')['Weekly_Sales'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=holi_wk_sales_no.index,
        y=holi_wk_sales_no.values,
        mode='lines',
        name="Not Holiday Week",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=holi_wk_sales_yes.index,
        y=holi_wk_sales_yes.values,
        mode='markers',
        name="Holiday Week",
        marker=dict(color='magenta')
    ))
    fig.update_layout(
        title='Weekly Sales During Holidays vs Non-Holidays',
        xaxis_title='Week Number',
        yaxis_title='Average Weekly Sales',
        legend_title='Week Type',
        template='plotly_dark'  # Apply the custom dark template
    )
    st.plotly_chart(fig)

    st.write("### Average Weekly Sales per Week (2010-2012)")
    if st.button("Show Weekly Sales Plot (2010-2012)"):
        wk_sales_2010 = train_merged[train_merged['Year'] == 2010].groupby('Week')['Weekly_Sales'].mean()
        wk_sales_2011 = train_merged[train_merged['Year'] == 2011].groupby('Week')['Weekly_Sales'].mean()
        wk_sales_2012 = train_merged[train_merged['Year'] == 2012].groupby('Week')['Weekly_Sales'].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wk_sales_2010.index, y=wk_sales_2010.values, mode='lines', name='2010'))
        fig.add_trace(go.Scatter(x=wk_sales_2011.index, y=wk_sales_2011.values, mode='lines', name='2011'))
        fig.add_trace(go.Scatter(x=wk_sales_2012.index, y=wk_sales_2012.values, mode='lines', name='2012'))
        fig.update_layout(
            title='Average Weekly Sales per Week (2010-2012)',
            xaxis_title='Week Number',
            yaxis_title='Average Weekly Sales ($)',
            legend_title='Year'
        )
        st.plotly_chart(fig)


elif selected_option == "💬 Chatbot":
    st.subheader("💬 Chatbot")
    st.write("Ask a question about the features of this app!")
    user_input = st.text_input("Enter your question:")
    if st.button("Ask"):
        matched_features = [key for key in feature_explanations if key.lower() in user_input.lower()]
        if matched_features:
            for feature in matched_features:
                st.write(f"**{feature}**: {feature_explanations[feature]}")
        else:
            st.write("I'm sorry, I couldn't find any relevant features in your question. Please try again!")


elif selected_option == "📤 Upload CSV":
    st.subheader("📤 Upload CSV Files")
    uploaded_file = st.file_uploader("Choose a CSV file to upload and make predictions", type=["csv"])
    
    if uploaded_file is not None:  # Check if a file has been uploaded
        try:
        
            uploaded_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data (Preview):")
            st.write(uploaded_data.head())
      
            required_features = [
                "Rolling_7_day_avg", "Lag_1_Week", "Lag_2_Weeks", "Lag_3_Weeks",
                "Dept_Avg_Sales", "Store_Avg_Sales", "Size", "Holiday_Impact",
                "MarkDown5", "Week_Sin", "Week_Cos", "Rolling_30_day_avg",
                "Sales_Per_Sqft", "Lag_4_Weeks"
            ]
        
            if not all(feature in uploaded_data.columns for feature in required_features):
                missing_features = [feature for feature in required_features if feature not in uploaded_data.columns]
                st.error(f"The uploaded file is missing the following required features: {', '.join(missing_features)}")
            else:
                prediction_data = uploaded_data[required_features]

                transformed_data = preprocessor.transform(prediction_data)
                predictions = model.predict(transformed_data)

                uploaded_data["Predicted_Weekly_Sales"] = predictions
                st.write("Predictions (Preview):")
                st.write(uploaded_data.head())

                csv = uploaded_data.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")


elif selected_option == "🚪 Exit":
    st.subheader("🚪 Exit")
    st.success("Thank you for using the application! 🤗")
    st.markdown("Hope to see you again! 🤖")
    st.balloons()

