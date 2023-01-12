#
FROM python:3.9

# Configure Poetry
ENV POETRY_VERSION=1.2.0
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"


WORKDIR /parrot-server

# Copy poetry files
COPY ./poetry.toml /parrot-server/poetry.toml
COPY ./poetry.lock /parrot-server/poetry.lock
RUN poetry install --no-interaction --no-cache --without dev

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./src /parrot-server/src

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]