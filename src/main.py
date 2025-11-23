import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("../data/student_stress.csv")

# Encode labels
le = LabelEncoder()
data["StressLevel"] = le.fit_transform(data["StressLevel"])

# Features & target
X = data.drop("StressLevel", axis=1)
y = data["StressLevel"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# User Input
print("\nEnter student details to predict stress level:")
sleep = float(input("Sleep Hours: "))
study = float(input("Study Hours: "))
screen = float(input("Screen Time (hours): "))
physical = float(input("Physical Activity (hours): "))
attendance = float(input("Attendance (%): "))

user_data = [[sleep, study, screen, physical, attendance]]
prediction = model.predict(user_data)[0]

stress_output = le.inverse_transform([prediction])[0]
print("\nPredicted Stress Level:", stress_output)