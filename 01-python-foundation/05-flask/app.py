from flask import Flask, render_template, request, redirect, url_for

"""
It creates and instance of the Flask class,
which will be the WSGI (Web Server Gateway Interface) application
We are passing in the entry point
So we are passing in __name__
"""
app = Flask(__name__)
@app.route("/",methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/about")
def sayHello():
    return render_template("about.html")

@app.route("/form",methods=['GET','POST'])
def form():
    if request.method == 'POST':
        # print("****************")
        # print("Request object:", request)
        # print("Request headers:", request.headers)
        # print("Request method:", request.method)
        # print("Request URL:", request.url)
        # print("Request form data:", request.form)
        # print("Request values:", request.values)
        name = request.form['hero']
        return f"Hello {name}"
    return render_template("form.html")

#Variable Rule <score> will replace it with a placeholder value which is a string. but we can make them into a particular data type by spedifying <int>

@app.route('/success/<int:score>')
def success(score:int):
    if score > 45:
        result = 'PASS'
        return render_template("result.html",results=result)
    result = "FAIL"
    return render_template("result.html",results=result)

@app.route("/success-results/<int:score>")
def successRes(score:int):
    res=''
    if score>=50:
        res="PASSED"
    else:
        res="FAILED"
    exp = {'score':score, "result":res}

    return render_template("result.html",results=exp)

@app.route('/submit',methods=('GET','POST'))
def calculate_total():
    if request.method == 'POST':
        math = float(request.form['maths'])
        biology = float(request.form['biology'])
        english = float(request.form['english'])
        total_score = (biology+english+math)/ 3
        return redirect(url_for("successRes",score=total_score)) # pass in the end point function and the parameter

    
    return render_template("marks.html")


if __name__ == "__main__":
    app.run(port=8000,debug=True)
