install:
	python -m pip install -r requirements.txt

ingest:
	python src/data_pipeline.py

train:
	python src/train.py

app:
	streamlit run app/streamlit_app.py
