DIR := ${CURDIR}

all:
	mvn compile assembly:single
	mvn test-compile

exp_speed:
	java -Dproject_home=${DIR} \
	-classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
	-Xmx8g -Xms1g edu.illinois.quicksel.experiments.SpeedComparison

exp_dmv:
	java -Dproject_home=${DIR} \
	-classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
	-Xmx32g -Xms1g edu.illinois.quicksel.experiments.DMVSpeedComparison

exp_job:
	java -Dproject_home=${DIR} \
	-classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
	-Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison \
	/home/jintao/CardinalityEstimationBenchmark/quicksel/test/java/edu/illinois/quicksel/resources/JOB/cols-sql/8/ 5000

exp_instacart:
	java -Dproject_home=${DIR} \
	-classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
	-Xmx32g -Xms1g edu.illinois.quicksel.experiments.InstacartSpeedComparison

exp_scan:
	java -Dproject_home=${DIR} \
	-classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar \
    -Xmx32g -Xms1g edu.illinois.quicksel.experiments.PerAttAndSamplingTest
