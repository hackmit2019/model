FROM python:3.7-slim-stretch
RUN pip install nltk numpy torch sklearn
RUN mkdir -p app
ADD call_clustering.py app/
ADD encoder app/encoder
ADD main.py app/
ADD models.py app/
ADD extract_features.py app/
CMD ["python", "app/main.py"]
