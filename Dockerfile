FROM python:3.9.18-bookworm as build
LABEL build_date="2023/11/1"
LABEL maintainer="Zane Kissel, et. al"

WORKDIR /usr/app/

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install "ray[tune]"

COPY . .

CMD ["python3", "-m", "src.main"]

# docker build -t [owner/image-name:v] .
# with image: docker run -it --rm --name grasp [owner/image-name:v]