# QuickSel

This repository contains the code used for the QuickSel paper in a quite unorganized manner.
The QuickSel paper appeared at the SIGMOD conference in 2020. 
The manuscript is also available from [this arxiv link](https://arxiv.org/abs/1812.10568).



## Download and install

After cloning this repository, running `make all` will build the project. 
Java and Maven are used for this project; thus,
to build the project, you will need JDK and Apache Maven installed on your machine.



## Examples
java
We provide some examples to compare the performance of QuickSel to others. After project is built successfully, you can run:
1. `make exp_speed` to run the speed test on synthetic generated data.
1. `make exp_dmv` to run the test on DMV dataset. 
1. `make exp_instacart` to run the test on Instacart dataset. 
1. `make exp_scan` to run the test to compare with scan-based methods such as sampling and histogram. 



## Misc.

The repository includes some other test code we wrote in the process of working on QuickSel. For example, `src/cpp/qpsolver` includes JNI-based code for calling `cvxopt`'s Quadratic Programming (QP) solver. Since QuickSel analytically computes the solution, the JNI-based code is not currently needed. However, if you want to re-enable the QP feature, extra steps are required (which is not currently documented here).


## License

Everything under [Apache License ver 2.0](https://www.apache.org/licenses/LICENSE-2.0) including future contributions. If worried about your copyright, don't contribute here.

