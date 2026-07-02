from flask import Flask, request, render_template
from src.pipline.predict_pipline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        fixed_acidity = float(request.form.get("fixed_acidity"))
        volatile_acidity = float(request.form.get("volatile_acidity"))
        citric_acid = float(request.form.get("citric_acid"))
        residual_sugar = float(request.form.get("residual_sugar"))
        chlorides = float(request.form.get("chlorides"))
        free_sulfur_dioxide = float(request.form.get("free_sulfur_dioxide"))
        total_sulfur_dioxide = float(request.form.get("total_sulfur_dioxide"))
        density = float(request.form.get("density"))
        ph = float(request.form.get("ph"))   
        sulphates = float(request.form.get("sulphates"))
        alcohol = float(request.form.get("alcohol"))

        data = CustomData(
            fixed_acidity=fixed_acidity,
            volatile_acidity=volatile_acidity,
            citric_acid=citric_acid,
            residual_sugar=residual_sugar,
            chlorides=chlorides,
            free_sulfur_dioxide=free_sulfur_dioxide,
            total_sulfur_dioxide=total_sulfur_dioxide,
            density=density,
            ph=ph, 
            sulphates=sulphates,
            alcohol=alcohol
        )

        pred_df = data.get_data_as_dataframe()
        print(request.form)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('index.html', results=result[0] , form_data=request.form)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)