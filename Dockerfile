FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY sic4dvar_algos /app/src/sic4dvar_algos
COPY sic4dvar_classes /app/src/sic4dvar_classes
COPY sic4dvar_functions /app/src/sic4dvar_functions
COPY lib /app/src/lib
COPY sic4dvar_modules /app/src/sic4dvar_modules

COPY sic4dvar_param_confluence.ini /app/
COPY sic4dvar_params.py /app/src/
COPY sic4dvar.py /app/src/

VOLUME ["/app/input", "/app/output", "/app/logs"]

ENTRYPOINT ["python", "/app/src/sic4dvar.py"]