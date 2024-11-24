FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .

RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY lambda_function_files/ ${LAMBDA_TASK_ROOT}

CMD [ "handler.handler" ]