python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/4/test
python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/4/train
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/
export PATH=$JAVA_HOME/bin:$PATH
java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols4/ 5000

# print_errors
cd ./test/java/edu/illinois/quicksel/resources/
python print_errors.py --testpath cols-sql/4/
