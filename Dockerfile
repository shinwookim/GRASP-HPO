FROM python:3.9.18-bookworm as build
LABEL build_date="2023/11/1"
LABEL maintainer="Zane Kissel, et. al"

WORKDIR /usr/app/src/

COPY requirements.txt ../

COPY src/*.py .

RUN mkdir hpo/
COPY src/hpo/*.py hpo/

RUN mkdir hpo/grasp/
COPY src/hpo/grasp/*.py hpo/grasp/

RUN mkdir hpo/benchmark/
COPY src/hpo/benchmark/*.py hpo/benchmark/

RUN mkdir input/
COPY src/input/*.py input/

RUN mkdir input/processor/
COPY src/input/processor/*.py input/processor/

RUN mkdir output/
COPY src/output/*.py output/

RUN mkdir output/results/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r ../requirements.txt

WORKDIR /usr/app/
CMD ["python3", "-m", "src.main"]

# docker build -t [owner/image-name:v] .
# with image: docker run -it --rm --name grasp [owner/image-name:v]

# not working currently