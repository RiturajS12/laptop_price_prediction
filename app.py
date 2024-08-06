from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

model = joblib.load('Random_Regression.lb')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('new.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route("/submit_form", methods=['POST'])
def prediction():
    if request.method == "POST":
        try:
            screen_size = float(request.form['screenSize'])
            ram = int(request.form['ram'])
            weight = float(request.form['weight'])
            processor_speed = float(request.form['processorSpeed'])
            width = float(request.form['width'])
            height = float(request.form['height'])
            storage_capacity = int(request.form['storageCapacity'])
            gpu_code = int(request.form['gpuCode'])
            
            company = request.form["company"]
            type_ = request.form["type"]
            os = request.form["os"]
            generation = request.form["generation"]
            series = request.form["series"]
            
            companies = [
                'Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu',
                'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI',
                'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba',
                'Vero', 'Xiaomi'
            ]
            types = [
                '2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook',
                'Ultrabook', 'Workstation'
            ]
            oss = [
                'Android', 'Chrome OS', 'Linux', 'Mac OS X', 'No OS',
                'Windows 10', 'Windows 10 S', 'Windows 7', 'macOS'
            ]
            generations = [
                'Generation 1', 'Generation 3', 'Generation 5', 'Generation 7'
            ]
            series_list = [
                'Series 0', 'Series A10', 'Series A12', 'Series A4',
                'Series A6', 'Series A72', 'Series A8', 'Series A9'
            ]
            
            def encode_categorical(value, categories):
                return [1 if category == value else 0 for category in categories]
            
            company_encoded = encode_categorical(company, companies)
            type_encoded = encode_categorical(type_, types)
            os_encoded = encode_categorical(os, oss)
            generation_encoded = encode_categorical(generation, generations)
            series_encoded = encode_categorical(series, series_list)
            
            UNSEEN_DATA = [[
                screen_size, ram, weight, processor_speed, width, height,
                storage_capacity, gpu_code
            ] + company_encoded + type_encoded + os_encoded + generation_encoded + series_encoded]
            
            prediction = model.predict(UNSEEN_DATA)[0]
            
            return render_template('output.html', 
                                   screen_size=screen_size,
                                   ram=ram,
                                   weight=weight,
                                   processor_speed=processor_speed,
                                   width=width,
                                   height=height,
                                   storage_capacity=storage_capacity,
                                   gpu_code=gpu_code,
                                   company=company,
                                   type=type_,
                                   os=os,
                                   generation=generation,
                                   series=series,
                                   output=prediction)
        
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
