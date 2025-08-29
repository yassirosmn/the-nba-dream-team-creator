FROM python:3.10-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# To do: pip install package (once done)

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
