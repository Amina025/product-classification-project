import joblib

# Učitavanje modela
model = joblib.load("product_classifier.pkl")

# Unos proizvoda
product_name = input("Unesite naziv proizvoda: ")

# Predikcija
prediction = model.predict([product_name])

print("Predviđena kategorija:", prediction[0])