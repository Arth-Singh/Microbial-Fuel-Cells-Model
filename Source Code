import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import plotly.graph_objs as go
import plotly.express as px
import random
import json
from datetime import datetime, timedelta

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load or initialize user data
@st.cache_data
def load_user_data():
    try:
        with open('user_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save user data
def save_user_data(data):
    with open('user_data.json', 'w') as f:
        json.dump(data, f)

# Create or load user data
user_data = load_user_data()

class User:
    def __init__(self, name):
        self.name = name
        self.energy_usage = np.random.randint(10, 100)
        self.points = 0
        self.level = 1
        self.badges = []
        self.usage_history = []

    def update_usage(self, new_usage):
        savings = self.energy_usage - new_usage
        self.energy_usage = new_usage
        self.points += max(savings * 10, 0)  # 10 points per unit of energy saved
        self.usage_history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), new_usage))
        self.check_level_up()
        self.check_badges()

    def check_level_up(self):
        new_level = self.points // 1000 + 1
        if new_level > self.level:
            self.level = new_level
            st.success(f"Congratulations! You've reached level {self.level}!")

    def check_badges(self):
        if self.points >= 5000 and "Energy Master" not in self.badges:
            self.badges.append("Energy Master")
            st.success("You've earned the 'Energy Master' badge!")
        if len(self.usage_history) >= 10 and "Consistent Saver" not in self.badges:
            self.badges.append("Consistent Saver")
            st.success("You've earned the 'Consistent Saver' badge!")

@st.cache_data
def create_dataset(n_samples=10000):
    """Create a more complex simulated dataset for MFC optimization."""
    df = pd.DataFrame({
        'flow_rate': np.random.uniform(1.0, 5.0, n_samples),
        'microbial_concentration': np.random.uniform(0.1, 2.0, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'organic_load': np.random.uniform(0.5, 3.0, n_samples)
    })

    # Create a more complex relationship
    df['energy_output'] = (
        5 * df['flow_rate']**2 +
        10 * np.log(df['microbial_concentration']) +
        0.5 * df['temperature']**2 -
        2 * df['organic_load']**3 +
        np.random.normal(0, 2, n_samples)  # Add some noise
    )

    return df

def feature_engineering(X):
    """Perform feature engineering."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    feature_names = poly.get_feature_names_out(X.columns)
    X_new = pd.DataFrame(X_poly, columns=feature_names)

    # Add some custom features
    X_new['temp_organic_interaction'] = X['temperature'] * X['organic_load']
    X_new['flow_conc_ratio'] = X['flow_rate'] / (X['microbial_concentration'] + 1e-5)

    return X_new

def build_advanced_nn_model(input_dim):
    """Build an advanced neural network model."""
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

@st.cache_resource
def train_models():
    """Train and return the Random Forest and Neural Network models."""
    df = create_dataset()
    X = df.drop('energy_output', axis=1)
    y = df['energy_output']
    X_engineered = feature_engineering(X)
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)

    nn_model = build_advanced_nn_model(X_train_scaled.shape[1])
    early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
    nn_model.fit(
        X_train_scaled, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        verbose=0
    )

    return rf_model, nn_model, scaler, X.columns

def get_ai_recommendation(user):
    base_recommendation = np.random.uniform(0.7, 0.9)
    if user.level > 5:
        base_recommendation *= 0.9  # More aggressive savings for higher level users
    return base_recommendation

def update_leaderboard(users):
    return sorted(users, key=lambda x: x.points, reverse=True)

def plot_leaderboard(users):
    fig = go.Figure(data=[
        go.Bar(name='Points', x=[user.name for user in users], y=[user.points for user in users])
    ])
    fig.update_layout(title='Leaderboard', xaxis_title='Users', yaxis_title='Points')
    return fig

def plot_user_history(user):
    df = pd.DataFrame(user.usage_history, columns=['timestamp', 'usage'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig = px.line(df, x='timestamp', y='usage', title=f'{user.name}\'s Energy Usage History')
    return fig

def create_mfc_simulation():
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(30)]
    mfc_data = {
        'date': dates,
        'energy_output': [random.uniform(50, 150) for _ in range(30)],
        'efficiency': [random.uniform(0.6, 0.9) for _ in range(30)]
    }
    return pd.DataFrame(mfc_data)

def plot_mfc_simulation(mfc_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mfc_data['date'], y=mfc_data['energy_output'], name='Energy Output'))
    fig.add_trace(go.Scatter(x=mfc_data['date'], y=mfc_data['efficiency'], name='Efficiency', yaxis='y2'))
    fig.update_layout(
        title='MFC Simulation: Energy Output and Efficiency',
        yaxis=dict(title='Energy Output (W)'),
        yaxis2=dict(title='Efficiency', overlaying='y', side='right')
    )
    return fig

def main():
    st.title("Microbial Fuel Cell (MFC) Energy Output Predictor and Gamified Experience")

    # User Authentication (simplified)
    user_name = st.text_input("Enter your username")
    if user_name:
        if user_name not in user_data:
            user_data[user_name] = vars(User(user_name))
        current_user = User(user_name)
        current_user.__dict__.update(user_data[user_name])
    else:
        st.warning("Please enter a username to continue.")
        return

    # Main Navigation
    page = st.sidebar.radio("Navigate", ["Home", "MFC Predictor", "User Profile", "Leaderboard", "MFC Simulation", "Eco Quiz"])

    if page == "Home":
        st.header("Welcome to the MFC Energy Management System")
        st.write(f"Hello, {current_user.name}! You are currently at level {current_user.level}.")
        st.write(f"Your current energy usage: {current_user.energy_usage} kWh")
        st.write(f"Your total points: {current_user.points}")

        if current_user.badges:
            st.write("Your badges:", ", ".join(current_user.badges))

        ai_recommendation = get_ai_recommendation(current_user)
        recommended_usage = current_user.energy_usage * ai_recommendation
        st.write(f"AI Recommendation: Try to reduce your usage to {recommended_usage:.2f} kWh")

        new_usage = st.number_input("Enter your new energy usage", min_value=0.0, max_value=float(current_user.energy_usage), value=float(recommended_usage))

        if st.button("Update Usage"):
            current_user.update_usage(new_usage)
            user_data[user_name] = vars(current_user)
            save_user_data(user_data)
            st.success(f"Usage updated! New points: {current_user.points}")

        st.plotly_chart(plot_user_history(current_user))

    elif page == "MFC Predictor":
        st.header("MFC Energy Output Predictor")

        rf_model, nn_model, scaler, original_features = train_models()

        flow_rate = st.slider("Flow Rate", 1.0, 5.0, 3.0)
        microbial_concentration = st.slider("Microbial Concentration", 0.1, 2.0, 1.0)
        temperature = st.slider("Temperature", 15.0, 35.0, 25.0)
        organic_load = st.slider("Organic Load", 0.5, 3.0, 1.5)

        new_mfc_config = pd.DataFrame([[flow_rate, microbial_concentration, temperature, organic_load]],
                                      columns=original_features)
        new_mfc_config_engineered = feature_engineering(new_mfc_config)
        new_mfc_config_scaled = scaler.transform(new_mfc_config_engineered)

        rf_prediction = rf_model.predict(new_mfc_config_engineered)
        nn_prediction = nn_model.predict(new_mfc_config_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest Prediction", f"{rf_prediction[0]:.2f} watts")
        with col2:
            st.metric("Neural Network Prediction", f"{nn_prediction[0][0]:.2f} watts")

    elif page == "User Profile":
        st.header("User Profile")
        st.write(f"Name: {current_user.name}")
        st.write(f"Level: {current_user.level}")
        st.write(f"Points: {current_user.points}")
        st.write(f"Current Energy Usage: {current_user.energy_usage} kWh")
        st.write("Badges:", ", ".join(current_user.badges))
        st.plotly_chart(plot_user_history(current_user))

    elif page == "Leaderboard":
        st.header("Leaderboard")
        leaderboard = update_leaderboard([User(name) for name in user_data])
        st.plotly_chart(plot_leaderboard(leaderboard))

    elif page == "MFC Simulation":
        st.header("MFC Simulation")
        if 'mfc_data' not in st.session_state:
            st.session_state.mfc_data = create_mfc_simulation()

        st.plotly_chart(plot_mfc_simulation(st.session_state.mfc_data))

        if st.button("Generate New Simulation"):
            st.session_state.mfc_data = create_mfc_simulation()
            st.experimental_rerun()

    elif page == "Eco Quiz":
        st.header("Eco Quiz")
        questions = [
            {
                "question": "What does MFC stand for in our context?",
                "options": ["Microbial Fuel Cell", "Modern Fuel Converter", "Magnetic Field Conductor", "Micro Fusion Chamber"],
                "correct": "Microbial Fuel Cell"
            },
            {
                "question": "Which of these is NOT typically used as a substrate in MFCs?",
                "options": ["Glucose", "Acetate", "Uranium", "Wastewater"],
                "correct": "Uranium"
            },
            {
                "question": "What is the main advantage of MFCs over traditional power generation methods?",
                "options": ["Higher energy output", "Lower cost", "Direct electricity generation from organic matter", "Faster energy production"],
                "correct": "Direct electricity generation from organic matter"
            }
        ]

        score = 0
        for i, q in enumerate(questions):
            st.subheader(f"Question {i+1}")
            st.write(q["question"])
            user_answer = st.radio(f"Select your answer for question {i+1}:", q["options"], key=f"q{i}")
            if user_answer == q["correct"]:
                score += 1

        if st.button("Submit Quiz"):
            st.write(f"Your score: {score}/{len(questions)}")
            points_earned = score * 50
            current_user.points += points_earned
            user_data[user_name] = vars(current_user)
            save_user_data(user_data)
            st.success(f"Quiz submitted! You've earned {points_earned} points. Your total points are now {current_user.points}.")

    # Save user data periodically to ensure updates
    save_user_data(user_data)

if __name__ == "__main__":
    main()
