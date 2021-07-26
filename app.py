import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template,flash,request,redirect,url_for,jsonify
from functions import load_model,show_inference

app = Flask(__name__)

app.secret_key = "sahilsc"

class ClientApp:
    def __init__(self):
        self.img_path = "inputs/input_img.png"
        self.model = load_model()

@app.route("/")
def home(): 
    return render_template('home.html',title="Vehicle Object Detection")

@app.route("/",methods=['POST'])
def predict(): 
    if request.method == 'POST' and 'imag' in request.files:
        img = request.files['imag']
        
        if img.filename=='':
            flash('Image not found')
            return redirect(url_for('home'))

        else:
            if img and img.filename.rsplit('.',1)[1].lower() in ['jpg','jpeg','png']:
                # Saving the image
                # img.save('static/'+img.filename)
                #read image file string data
                filestr = img.read()
                #convert string data to numpy array
                npimg = np.frombuffer(filestr, np.uint8)
                # convert numpy array to image
                img_vec = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
                # prediction
                pred_img = show_inference(capp.model,img_vec)
                # saving the prediction
                pre_img_fname = 'static/predictions/prediction.jpg'
                cv2.imwrite(pre_img_fname,pred_img)

                flash('Succesfully made the predictions')
                return render_template('home.html',title="Vehicle Object Detection",filename="predictions/prediction.jpg")
           
            
            else:
                flash('Invalid Image format')
                return redirect(url_for('home'))

    else:
        return redirect(url_for('home'))

@app.route('/api',methods=['POST'])
def api():
    img = request.files['imag']
    filestr = img.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img_vec = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
    pred_img = show_inference(capp.model,img_vec)
    pre_img_fname = 'static/predictions/prediction.jpg'
    cv2.imwrite(pre_img_fname,pred_img)
    with open(pre_img_fname, "rb") as f:
        b64_img = base64.b64encode(f.read())
    return jsonify({"image" : b64_img.decode('utf-8')})


    # return 'inside predict'
    # return render_template('home.html',title="Vehicle Object Detection")








if __name__ == '__main__':
    capp = ClientApp()
    app.run(debug=True)