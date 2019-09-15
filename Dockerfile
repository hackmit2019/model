FROM python:3.7-slim-stretch
RUN pip install nltk numpy torch sklearn spacy
RUN mkdir -p app
ADD call_clustering.py app/
ADD encoder app/encoder
ADD main.py app/
ADD models.py app/
ADD extract_features.py app/
WORKDIR app/
RUN python -m spacy download en_core_web_sm
CMD ["python", "main.py"]
