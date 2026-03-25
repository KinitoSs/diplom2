FROM apache/spark:4.1.1

USER root
RUN pip install --no-cache-dir pyspark \
    && curl -L -o /opt/spark/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    && curl -L -o /opt/spark/jars/aws-java-sdk-bundle-1.12.536.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.536/aws-java-sdk-bundle-1.12.536.jar
