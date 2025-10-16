FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=agent
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid "$USER_GID" "$USERNAME" \
    && useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME"

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . /workspace

RUN chown -R "$USER_UID:$USER_GID" /workspace

ENV PYTHONPATH=/workspace/src

USER $USERNAME

ENTRYPOINT ["/bin/bash"]
