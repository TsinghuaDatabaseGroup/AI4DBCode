python preprocessing.py --raw-file ../train-test-data/forest_power-sql/foresttest.sql --min-max-file ../learnedcardinalities-master/data/forest_min_max_vals.csv --datasets-dir ../train-test-data/forest_power-data/forest --output-dir JOB/forest/test
python preprocessing.py --raw-file ../train-test-data/forest_power-sql/foresttrain.sql --min-max-file ../learnedcardinalities-master/data/forest_min_max_vals.csv --datasets-dir ../train-test-data/forest_power-data/forest --output-dir JOB/forest/train
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/
export PATH=$JAVA_HOME/bin:$PATH
java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/forest/ 5000

# To modify the path of a Print Error

cd ./test/java/edu/illinois/quicksel/resources/
python print_errors.py --testpath ./forest/
