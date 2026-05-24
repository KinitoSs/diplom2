FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    tokenizers==0.15.0 \
    safetensors==0.4.0

RUN pip install --no-cache-dir \
    accelerate==0.25.0 \
    bitsandbytes==0.41.0 \
    peft==0.7.0

RUN pip install --no-cache-dir \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    tensorboard==2.14.0

RUN pip install --no-cache-dir \
    opencv-python-headless \
    albumentations \
    datasets

RUN pip install --no-cache-dir \
    "wget>=3.2" \
    "tqdm>=4.66.0" \
    "pillow>=10.0.0" \
    "numpy>=1.24.0" \
    jupyterlab \
    ipykernel \
    fiftyone \
    pycocotools


RUN pip install --no-cache-dir --force-reinstall "numpy<2"
RUN pip install --no-cache-dir --upgrade \
    packaging \
    setuptools

COPY train.py .

EXPOSE 8888

# CMD ["python", "train.py"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]