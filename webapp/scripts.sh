# Generate env
pip install virtualenv
virtualenv sentiment-flask-app

# Activate env
source sentiment-flask-app/bin/activate

# Install reqs
pip3 install flask gevent requests pillow tensorflow keras

# Start flask server
python3 main.py
