FROM continuumio/miniconda3

WORKDIR /app

COPY enviroment.yml .

RUN conda env create -f enviroment.yml

ENV PATH /opt/conda/envs/AA1GPU/bin:$PATH

COPY . .

CMD ["python", "inference.py"]