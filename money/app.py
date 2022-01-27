# flask web server

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register():
    return "register"


@app.route("/main")
def main_page():
    return "main page"


if __name__ == "__main__":
    app.run(debug=True)