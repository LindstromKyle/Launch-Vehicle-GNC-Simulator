FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workdir

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /workdir/logs /workdir/src/mc_results \
    && chown -R appuser:appuser /workdir
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
