import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
parser.add_argument('--xt', type=int, help='datasets_dir', default=4)
args = parser.parse_args()
cols = args.cols
xt = args.xt

if cols == 2:
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/col2_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/2/test')
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --min-max-file ../learnedcardinalities-master/data/col2_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/2/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols2/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/2/')

elif cols == 4:
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/4/test')
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/4/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols4/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/4/')

elif cols == 6:
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/col6_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/6/test')
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --min-max-file ../learnedcardinalities-master/data/col6_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/6/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols6/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/6/')

elif cols == 8:
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/col8_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/8/test')
    os.system(
        'python preprocessing.py --raw-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --min-max-file ../learnedcardinalities-master/data/col8_min_max_vals.csv --datasets-dir ../train-test-data/imdbdata-num/ --output-dir JOB/cols-sql/8/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols8/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/8/')

elif xt == 2:
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol2_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/2/test')
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/2/train-2-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol2_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/2/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols2/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/2/')

elif xt == 4:
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol4_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/4/test')
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/4/train-4-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol4_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/4/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols4/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/4/')

elif xt == 6:
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol6_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/6/test')
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/6/train-6-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol6_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/6/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols6/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/6/')

elif xt == 8:
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol8_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/8/test')
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/8/train-8-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol8_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/8/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols8/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/8/')

elif xt == 10:
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol10_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/10/test')
    os.system(
        'python preprocessing_xtzx.py --raw-file ../train-test-data/xtzx-data-sql/10/train-10-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol10_min_max_vals.csv --datasets-dir ../train-test-data/xtzx-data-sql/ --output-dir JOB/cols-sql/10/train')
    os.system('export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_40/')
    os.system('export PATH=$JAVA_HOME/bin:$PATH')
    os.system(
        'java -classpath target/test-classes:target/quickSel-0.1-jar-with-dependencies.jar -Xmx32g -Xms1g edu.illinois.quicksel.experiments.JOBSpeedComparison ./test/java/edu/illinois/quicksel/resources/JOB/cols10/ 5000')
    os.system('cd ./test/java/edu/illinois/quicksel/resources/')
    os.system('python print_errors.py --testpath cols-sql/10/')
