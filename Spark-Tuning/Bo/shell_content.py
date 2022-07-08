#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zzz_jq
# Time: 2020/10/2 18:13
front = ["""#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/ConnectedComponentApp-1.0.jar"
CLASS="src.main.scala.ConnectedComponentApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${numV}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""",
"""#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`

echo "========== running ${APP} benchmark =========="

DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"



DU ${INPUT_HDFS} SIZE 
JAR="${DIR}/target/DecisionTreeApp-1.0.jar"
CLASS="DecisionTree.src.main.java.DecisionTreeApp"


setup
for((i=0;i<${NUM_TRIALS};i++)); do		
	# classification
	RM ${OUTPUT_HDFS_Classification}
	purge_data "${MC_LIST}"	
	OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS_Classification} ${NUM_OF_CLASS_C} ${impurityC} ${maxDepthC} ${maxBinsC} ${modeC}"
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""", """#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} bench =========="


# pre-running
DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/KMeansApp-1.0.jar"
CLASS="KmeansApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_CLUSTERS} ${MAX_ITERATION} ${NUM_RUN}"


setup
set_gendata_opt
for((i=0;i<${NUM_TRIALS};i++)); do
    RM ${OUTPUT_HDFS}
    # (Optional procedure): free page cache, dentries and inodes.
    # purge_data "${MC_LIST}"
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`

""", """
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 
[ -z $SIZE ] && SIZE=0
JAR="${DIR}/target/LabelPropagationApp-1.0.jar"
CLASS="src.main.scala.LabelPropagationApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${numV} ${NUM_OF_PARTITIONS}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`
""", """#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} workload =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/LinearRegressionApp-1.0.jar"
CLASS="LinearRegression.src.main.java.LinearRegressionApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${MAX_ITERATION} "

setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM  ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
	START_TS=`get_start_ts`
	START_TIME=`timestamp`

""", """#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} workload =========="


DU ${INPUT_HDFS} SIZE 
CLASS="LogisticRegression.src.main.java.LogisticRegressionApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS}  ${MAX_ITERATION} ${STORAGE_LEVEL} "

JAR="${DIR}/target/LogisticRegressionApp-1.0.jar"

setup
for((i=0;i<${NUM_TRIALS};i++)); do
    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;

    START_TIME=`timestamp`

""",
"""#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check
DU ${INPUT_HDFS} SIZE 

#JAR="${DIR}/target/scala-2.10/pagerankapp_2.10-1.0.jar"
JAR="${DIR}/target/PageRankApp-1.0.jar"
CLASS="src.main.scala.pagerankApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${MAX_ITERATION} ${TOLERANCE} ${RESET_PROB} ${STORAGE_LEVEL}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	echo "${APP} opt ${OPTION}"
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""", """#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`

DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} bench =========="

# pre-running
DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/PCAApp-1.0.jar"
CLASS="PCA.src.main.scala.PCAApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${DIMENSIONS}"

setup
for((i=0;i<${NUM_TRIALS};i++)); do
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`

""", """#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/PregelOperationApp-1.0.jar"
CLASS="src.main.scala.PregelOperationApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do

    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`

""", """#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/ShortestPathsApp-1.0.jar"
CLASS="src.main.scala.ShortestPathsApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${numV}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""", """#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/StronglyConnectedComponentApp-1.0.jar"
CLASS="src.main.scala.StronglyConnectedComponentApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""",
"""
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/SVDPlusPlusApp-1.0.jar"
CLASS="src.main.scala.SVDPlusPlusApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${NUM_ITERATION} ${RANK} ${MINVAL} ${MAXVAL} ${GAMMA1} ${GAMMA2} ${GAMMA6} ${GAMMA7} ${STORAGE_LEVEL}"

echo "opt ${OPTION}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do

    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`

""",
"""#!/bin/bash


bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== Running ${APP} Workload =========="

DU ${INPUT_HDFS} SIZE 
CLASS="SVM.src.main.java.SVMApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${MAX_ITERATION} ${STORAGE_LEVEL}"
JAR="${DIR}/target/SVMApp-1.0.jar"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	# path check	
	RM ${OUTPUT_HDFS}
START_TS=`get_start_ts`;
	START_TIME=`timestamp`

""", """
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check
DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/TerasortApp-1.0-jar-with-dependencies.jar"
CLASS="src.main.scala.terasortApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} "
Addition_jar="--jars ${DIR}/target/jars/guava-19.0-rc2.jar"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
    echo "${APP} opt ${OPTION}"
    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`
""",
"""
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="

# path check
DU ${INPUT_HDFS} SIZE 
JAR="${DIR}/target/TriangleCountApp-1.0.jar"
CLASS="src.main.scala.triangleCountApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${STORAGE_LEVEL}"
echo "opt ${OPTION}"

setup
for((i=0;i<${NUM_TRIALS};i++)); do
    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;
    START_TIME=`timestamp`

"""
         ]




#LinearRegression, LogisticRegression, MatrixFactorization, PageRank
front1 = ["""#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} workload =========="


# path check

DU ${INPUT_HDFS} SIZE 

JAR="${DIR}/target/LinearRegressionApp-1.0.jar"
CLASS="LinearRegression.src.main.java.LinearRegressionApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${MAX_ITERATION} "

setup
for((i=0;i<${NUM_TRIALS};i++)); do
	
	RM  ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
	START_TS=`get_start_ts`
	START_TIME=`timestamp`
""", '''
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} workload =========="


DU ${INPUT_HDFS} SIZE 
CLASS="LogisticRegression.src.main.java.LogisticRegressionApp"
OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS}  ${MAX_ITERATION} ${STORAGE_LEVEL} "

JAR="${DIR}/target/LogisticRegressionApp-1.0.jar"

setup
for((i=0;i<${NUM_TRIALS};i++)); do
    RM ${OUTPUT_HDFS}
    purge_data "${MC_LIST}"	
    START_TS=`get_start_ts`;

    START_TIME=`timestamp`
''',  '''
#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} benchmark =========="


# path check
DU ${INPUT_HDFS} SIZE 

#JAR="${DIR}/target/scala-2.10/pagerankapp_2.10-1.0.jar"
JAR="${DIR}/target/PageRankApp-1.0.jar"
CLASS="src.main.scala.pagerankApp"
OPTION="${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_PARTITIONS} ${MAX_ITERATION} ${TOLERANCE} ${RESET_PROB} ${STORAGE_LEVEL}"


setup
for((i=0;i<${NUM_TRIALS};i++)); do
	echo "${APP} opt ${OPTION}"
	RM ${OUTPUT_HDFS}
	purge_data "${MC_LIST}"	
START_TS=`get_start_ts`;
	START_TIME=`timestamp`
''']

rear = """    
res=$?;
	END_TIME=`timestamp`
get_config_fields >> ${BENCH_REPORT}
print_config  ${APP} ${START_TIME} ${END_TIME} ${SIZE} ${START_TS} ${res}>> ${BENCH_REPORT};
done
teardown
exit 0
"""
