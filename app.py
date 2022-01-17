from flask import Flask, render_template, request, redirect
from speech_recognizer import speech_recog
import random, os


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_name = str(random.randint(0, 100000))
            file.save(file_name)

            # invoke keyword spotting service
           
            # make a prediction
            transcript = speech_recog(file_name)
            print
            # remove the audio file
            os.remove(file_name)

    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
