import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("project1.csv")

print("First 5 rows:")
print(df.head())

print("Dataset shape:")
print(df.shape)

print("Data information:")
print(df.info())

print("Missing value:")
print(df.isnull().sum())

print("Statistical summary:")
print(df.describe())


x = df[["Hours", "Attendance", "PrevMarks"]]
y = df["Result"]

print("Feature:")
print(x.head())

print("Target:")
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(
 x, y, test_size=0.2, random_state=42)

print("Traninig data: ",X_train.shape)
print("Testing data:" , X_test.shape)

print("Training Target:" , y_train.shape)
print("Testing Target:" , y_test.shape)


scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

print("Scaled Training Data:")
print(X_train[:5])

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train , y_train)

print("KNN Model Trained Succesfully")

y_pred = model.predict(X_test)

print("\nActual Results:")
print(y_test.values[:10])

print("\nPredicted Results:")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")



joblib.dump({"model" : model , "Scalar" : scalar},
"KNN_model.pkl")

print("Model Saved Succesfull")
