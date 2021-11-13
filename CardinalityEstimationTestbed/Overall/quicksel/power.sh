python preprocessing.py --raw-file ../train-test-data/forest_power-sql/powertest.sql --min-max-file ../learnedcardinalities-master/data/power_min_max_vals.csv --datasets-dir ../train-test-data/forest_power-data/power --output-dir JOB/power/test
python preprocessing.py --raw-file ../train-test-data/forest_power-sql/powertrain.sql --min-max-file ../learnedcardinalities-master/data/power_min_max_vals.csv --datasets-dir ../train-test-data/forest_power-data/power --output-dir JOB/power/train
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/
export PATH=$JAVA_HOME/bin:$PATH
java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/power/ 5000

# To modify the path of a Print Error

cd ./test/java/edu/illinois/quicksel/resources/
python print_errors.py --testpath ./power/
