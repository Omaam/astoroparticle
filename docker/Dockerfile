# FROM python:3.9
FROM tensorflow/tensorflow:2.13.0-gpu

ENV USER_NAME="user"
RUN apt-get update && apt-get install -y \
  vim \
  zsh \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
  && pip install --no-cache \
    astropy \
    matplotlib \
    numpy \
    seaborn \
    scipy \
    # tensorflow==2.13.* \
    tensorflow-probability==0.21.*

RUN useradd -m -s /bin/zsh ${USER_NAME} \
  && chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

ENV SHELL="/bin/zsh"

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD [ "zsh" ]
