FROM python:3.9.18-bookworm as build
LABEL build_date="20/10/2023"
LABEL maintainer="Zane Kissel, et. al"

WORKDIR /usr/app/src/

RUN mkdir main/
COPY src/main/*.py main/
COPY requirements.txt ../

# tests and test output directory
RUN mkdir test/
COPY src/test/test.sh test/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r ../requirements.txt

CMD ["test/test.sh"]

# docker build -t [owner/image-name:v] .
# with image: docker run -it --rm --name grasp [owner/image-name:v]
# run ./test/test.sh once in container