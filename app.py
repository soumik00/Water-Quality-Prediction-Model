from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load and preprocess data
df = pd.read_csv("water_potability.csv")
x = df.drop("Potability", axis=1)
y = df.Potability
scaler = StandardScaler()
# x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Train your model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,shuffle=True,random_state=42)
model_RFC = RandomForestClassifier(n_estimators=10)
parameters = {
    'n_estimators': [1000],
    'criterion': ['log_loss'],  # Using entropy for classification
    'max_features': ['sqrt'],
    'n_jobs': [-1]
}
R_F_C_G_CV = GridSearchCV(estimator=model_RFC, param_grid=parameters, cv=20)
model_RFC.fit(x_train,y_train)



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def font():
    if request.method == 'POST':
        ph1 = request.form.get('_ph')
        ph = float(ph1)

        Hardness1 = request.form.get('_hardness')
        Hardness = float(Hardness1)

        Solids1 = request.form.get('_Solids')
        Solids = float(Solids1)


        Chloramines1 = request.form.get('_Chloramines')
        Chloramines = float(Chloramines1)
        Sulfate1 = request.form.get('_Sulfate') 
        Sulfate = float(Sulfate1)
        Conductivity1 = request.form.get('_Conductivity')
        Conductivity = float(Conductivity1)  # Added for missing field
        Organic_carbon1 = request.form.get('_Organic_carbon')
        Organic_carbon = float(Organic_carbon1)
        Trihalomethanes1 = request.form.get('_Trihalomethanes')
        Trihalomethanes = float(Trihalomethanes1)
        Turbidity1 = request.form.get('_Turbidity')
        # Potability = request.form.get('_Potability')  # Added for missing field
        Turbidity = float(Turbidity1)
        user_input_features = [[ph,
Hardness,
Solids,
Chloramines,
Sulfate,
Conductivity,
Organic_carbon,
Trihalomethanes,
Turbidity]]
        prediction = model_RFC.predict(user_input_features)
        
        # Return prediction result
        score_RFC_test =  model_RFC.score(x_test,y_test)
        message = score_RFC_test
        print(prediction)
        if prediction[0] == 0:
            return render_template('bad_water.html',message=message)
        else:
            return render_template('good_water.html',message=message) 
    
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
