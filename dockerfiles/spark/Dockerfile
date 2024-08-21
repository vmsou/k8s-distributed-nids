FROM bitnami/spark:3.5.1
USER root
RUN install_packages curl
RUN pip install --no-cache numpy
COPY /apps/* /opt/bitnami/spark/apps/
USER 1001
RUN curl https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.13/3.5.1/spark-sql-kafka-0-10_2.13-3.5.1.jar --output /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.13-3.5.1.jar