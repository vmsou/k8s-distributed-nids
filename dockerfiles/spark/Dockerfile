FROM bitnami/spark:3.5.1
USER root
RUN pip install --no-cache numpy
RUN apt-cache search openblas
RUN install_packages curl libopenblas-dev
RUN apt-get clean

COPY /apps/* /opt/bitnami/spark/apps/
COPY /bin/* /opt/bitnami/spark/bin/
RUN chmod +x /opt/bitnami/spark/bin/*

USER 1001
RUN curl https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar --output /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.1.jar
RUN curl https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.1/kafka-clients-3.5.1.jar --output /opt/bitnami/spark/jars/kafka-clients-3.5.1.jar
RUN curl https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar --output /opt/bitnami/spark/jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar
RUN curl https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar --output /opt/bitnami/spark/jars/commons-pool2-2.11.1.jar
