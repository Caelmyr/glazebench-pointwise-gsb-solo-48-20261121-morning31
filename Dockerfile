ARG GLAZEBENCH_DEPS_IMAGE=python:3.11-slim
FROM ${GLAZEBENCH_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/glazebench
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 GLAZEBENCH_DATA=/var/lib/glazebench
WORKDIR /opt/glazebench
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY glazebench ./glazebench
COPY static ./static
RUN useradd -U -u 11501 glazebench && install -d -o glazebench -g glazebench /var/lib/glazebench
USER glazebench
EXPOSE 8000
CMD ["python", "-m", "glazebench"]
