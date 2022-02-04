# flask web server

from flask import Flask, render_template, request
import json

app = Flask(__name__)


@app.route("/")
def home():
    if request.method == "POST":
        name = request.args["user_id"]
    return render_template("home.html")


@app.route("/login", methods=["POST"])
def login_method():
    if request.method == "POST":
        name = request.form["user_id"]
        passwd = request.form["user_pw"]
        sql = f"select * from test where user_id = {name}, user_pw = {passwd}"
        response = {
            "results": f"{sql}"
        }
        return json.dumps(response)


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/main")
def main_page():
    return render_template("main.html")


@app.route("/manager")
def manager_page():
    return render_template("manager.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)