FROM python:3.7-slim

RUN pip install tensorflow==2.6.0
CMD python -c "import tensorflow as tf; print(tf.constant(42) / 2 + 2)"