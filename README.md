everything is inside "sentiment" 

# install after clone 
pip install -r requirements.txt 

# run server 
python manage.py runserver

# train model 
python manage.py train_model

# analyse tweet: 
http://127.0.0.1:8000/sentiment/type/
# analyse tweets about a subject:
http://127.0.0.1:8000/sentiment/search/
