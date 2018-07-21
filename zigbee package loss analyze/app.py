from flask import Flask, request, redirect, url_for, render_template
from redis import Redis, RedisError
import os
import socket
from werkzeug.utils import secure_filename
import os
import analyze

UPLOAD_FOLDER = '/app/'
ALLOWED_EXTENSIONS = set(['pcap', 'pcapng'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part!"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return "No file selected!"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_url)
            cmd = 'tshark -r ' + file_url + " -T fields -e frame.time_epoch -e wpan.frame_type -e wpan.seq_no -e wpan.dst_pan -e wpan.dst16 -e wpan.src16 > /app/zigbee.output"
            print cmd
            os.system(cmd)
            cmd = "rm " + file_url
            os.system(cmd)
            result = analyze.process_file("/app/zigbee.output")
            return render_template('result_template.html',filename=filename, result=result)
    
    return '''
    <!doctype html>
    <title>Upload Zigbee Pacap file to Analyze</title>
    <h1>Upload Zigbee Pacap file to Analyze</h1>
    <form method=post enctype=multipart/form-data>
      Upload File:<br>
      <br>
      <input type=file name=file>
      <br>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
