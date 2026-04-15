FROM python:3.11-bookworm as base-os

ARG POETRY_HTTP_BASIC_ML_TOOLS_USERNAME \
    POETRY_HTTP_BASIC_ML_TOOLS_PASSWORD

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHON_INTERPRETER_PATH=/usr/local/bin/python \
    PATH="/root/.local/bin:$PATH"

RUN git config --global init.defaultBranch main

RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -

COPY pyproject.toml poetry.lock /

FROM base-os as base

RUN poetry install --without=dev --no-interaction --no-root && \
    rm -rf ~/.cache/pypoetry/artifacts ~/.cache/pypoetry/cache

FROM base as base-dev

RUN poetry install --no-interaction --no-root && \
    rm -rf ~/.cache/pypoetry/artifacts ~/.cache/pypoetry/cache

FROM base-dev as development

ENV ENVIRONMENT=DEVELOPMENT

RUN poetry completions bash >> ~/.bash_completion

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"

FROM base-dev as testing

ENV APP_PATH=/app

WORKDIR $APP_PATH

ADD . $APP_PATH

ENV PYTHONPATH "${PYTHONPATH}:${APP_PATH}"

CMD pytest -p no:cov --junitxml /temp/result.xml -m "not integration and not notest" $APP_PATH
