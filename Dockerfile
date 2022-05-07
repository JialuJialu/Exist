# builder stage to install all requirements
FROM python:3.9-slim AS builder
WORKDIR /env
COPY requirements.txt /requirements.txt

# more info - https://testdriven.io/blog/django-docker-traefik/
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# create venv and install requirements
RUN python -m venv local_env && \
    . local_env/bin/activate && \
    /env/local_env/bin/python -m pip install --upgrade pip && \
    #apt-get update && apt-get install python3-dev libpq-dev gcc -y  && \
    pip install --no-cache-dir -r /requirements.txt



# runner stage to actually run application
FROM python:3.9-slim as runner
WORKDIR /project
# copy requirements from builder to runner
COPY --from=builder /env/local_env ./local_env
#COPY . .


RUN apt-get update && \
    . local_env/bin/activate && apt-get install xz-utils -y

# # include python in path at beginning
ENV PATH=/app/local_env/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD /bin/bash
# ENV APPLICATION_PORT=8501
# EXPOSE ${APPLICATION_PORT}
# CMD python -m uvicorn app:app --host=0.0.0.0 --port=${APPLICATION_PORT}