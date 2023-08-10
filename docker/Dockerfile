FROM python:3.9

ENV USER_NAME="user"
RUN apt-get update \
  && apt-get install -y zsh

RUN pip install --upgrade pip \
  && pip install tensorflow==2.13.0 \
  && pip install matplotlib \
  && pip install seaborn \
  && pip install tensorflow-probability==0.21.0

RUN useradd -m -s /bin/zsh ${USER_NAME} \
  && chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

ENV SHELL="/bin/zsh"

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD [ "zsh" ]
