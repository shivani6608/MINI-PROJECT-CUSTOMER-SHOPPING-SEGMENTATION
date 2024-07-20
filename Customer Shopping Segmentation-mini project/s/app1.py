from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__)
# Define possible values for each categorical feature
genders = ['Male', 'Female', 'Other']
payment_methods = ['Credit Card', 'Debit Card', 'Cash']
shopping_malls = ['Kanyon', 'Forum Istanbul', 'Metrocity', 'Metropol AVM', 'Istinye Park',
                  'Mall of Istanbul', 'Emaar Square Mall', 'Cevahir AVM', 'Viaport Outlet', 'Zorlu Center']

# Train label encoders
gender_le = LabelEncoder().fit(genders)
payment_method_le = LabelEncoder().fit(payment_methods)
shopping_mall_le = LabelEncoder().fit(shopping_malls)

# Save label encoders
label_encoders = {
    'gender': gender_le,
    'payment_method': payment_method_le,
    'shopping_mall': shopping_mall_le
}

joblib.dump(label_encoders, 'label_encoders5.pkl')


# Load the model and scaler
model = joblib.load('kmeans_model5.pkl')
scaler = joblib.load('scaler5.pkl')


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/shopping')
def shopping():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        # Validate incoming data
        required_fields = ['gender', 'payment_method', 'shopping_mall', 'age', 'quantity', 'price']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field: {field}")

        # Convert data into DataFrame
        df = pd.DataFrame([data])

        # Encoding categorical variables
        for column in ['gender', 'payment_method', 'shopping_mall']:
            if column in label_encoders:
                le = label_encoders[column]
                print(f"Label Encoder classes for {column}: {le.classes_}")
                if data[column] not in le.classes_:
                    raise ValueError(f"Value '{data[column]}' for column '{column}' not found in label encoder classes.")
                df[column] = le.transform([data[column]])
            else:
                raise ValueError(f"No label encoder found for column: {column}")

        # Feature scaling
        print(f"Data before scaling: {df[['age', 'quantity', 'price']]}")
        df[['age', 'quantity', 'price']] = scaler.transform(df[['age', 'quantity', 'price']])
        print(f"Data after scaling: {df[['age', 'quantity', 'price']]}")

        # Making predictions
        print(f"Data before prediction: {df}")
        prediction = model.predict(df)
        print(f"Predicted cluster: {prediction[0]}")

        return jsonify({'predicted_cluster': int(prediction[0])})

    except ValueError as ve:
        print(f"Validation Error: {ve}")
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
