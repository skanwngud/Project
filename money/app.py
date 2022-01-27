# flask web server

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


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