FROM nvcr.io/nvidia/pytorch:21.11-py3

ARG USERNAME=pytorch
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USER_GECOS=PyTorch

RUN \
  # Create a non-root user
  groupadd --gid $USER_GID $USERNAME \
  && useradd \
    -s /bin/bash \
    --uid $USER_UID \
    --gid $USER_GID \
    -c "$USER_GECOS" \
    -m $USERNAME \
  # Common packages
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    ghostscript \
    vim \
    sudo \
    locales \
    less \
  # Add sudo support for non-root user
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  # Ensure at least the en_US.UTF-8 UTF-8 locale is available.
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  # JP locale
  && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen \
  # Clean up
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV EDITOR=vim
