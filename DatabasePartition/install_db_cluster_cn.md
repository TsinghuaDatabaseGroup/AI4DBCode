# Greenplum安装

安装环境：centos7.5
安装版本：greenplum-6.21.0
安装机器：182.92.64.198（主机） 39.105.229.60（节点1）

### root用户，关闭防火墙
```
chkconfig iptables off
systemctl stop firewalld.service
systemctl disable firewalld.service
```
![](media/16593430874831.jpg)
### root用户，关闭SELINUX
```
vim /etc/selinux/config
```

```
# SELINUX=disabled
```
![](media/16593431072046.jpg)

### root用户，修改host，每个节点都操作
```
vim /etc/hosts
```
```
182.92.64.198    gpmaster
39.105.229.60   gpseg1
```

![](media/16593433130201.jpg)


### root用户，创建组和用户 (每个节点都操作)
```
groupadd -g 530 gpadmin
useradd -g 530 -u530 -m -d /home/gpadmin -s /bin/bash gpadmin
chown -R gpadmin:gpadmin /home/gpadmin/
passwd gpadmin
```

### 安装依赖库
```
sudo yum install apr apr-util bash bzip2 curl krb5 libcurl libevent \
libxml2 libyaml zlib openldap openssh openssl openssl-libs perl readline rsync R sed tar zip krb5-devel
```

```
vim /etc/security/limits.conf
```
```
* soft nofile 65536 
* hard nofile 65536
* soft nproc 131072
* hard nproc 131072
```

依次复制到其他节点
```
scp /etc/sysctl.conf gpseg1:/etc
scp /etc/security/limits.conf gpseg1:/etc/security
scp /etc/selinux/config gpseg1:/etc/selinux
```

最后让节点配置生效，在每个节点依次执行如下命令
```
sysctl -p
```

### root用户，上传程序包，提升文件权限
程序包下载地址：https://github.com/greenplum-db/gpdb/releases/download/6.21.0/open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm
上传或者下载到：`/home/gpadmin` 目录
```
scp -r open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm root@39.105.229.60:/home/gpadmin/
```
使用 sudo（或以 root 身份），使用系统的包管理器软件在所有主机上安装 Greenplum 数据库包
```
cd /home/gpadmin/
chmod +x open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm 
rpm -ivh open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm 
#默认安装到/usr/local，授权给gpadmin
chown -R gpadmin:gpadmin /usr/local
#设置环境变量
source /usr/local/greenplum-db/greenplum_path.sh
```

### 在区段服务器（gpseg1）安装GP软件
1) 建立互信
```
vi /home/gpadmin/all_hosts_file
```
```
gpmaster
gpseg1
```

建立gpadmin互信：(gpadmin操作)
```
su gpadmin
```
```
ssh-keygen
ssh-copy-id gpseg1
source /usr/local/greenplum-db/greenplum_path.sh 
gpssh-exkeys -f /home/gpadmin/all_hosts_file  
```

2) 为区段服务器创建用户，目录文件夹等

首先创建区段Host清单文件（gpadmin）
```
vi /home/gpadmin/seg_hosts_file
```
```
gpseg1
```


3) 实例初始化之前的准备工作
初始化前同步系统时钟（gpadmin）
```
1) 检查各区段服务器的系统时钟同步否

$ gpssh -f /home/gpadmin/all_hosts_file -v date

2) 同步各服务器的时钟

$ gpssh -f /home/gpadmin/all_hosts_file -v ntpd
```

4) 修改Greenplum数据库配置文件
```
cp /usr/local/greenplum-db-6.21.0/docs/cli_help/gpconfigs/gpinitsystem_config /home/gpadmin/gpinitsystem_config  //复制模板配置文件
```
```
vi  /home/gpadmin/gpinitsystem_config  //修改关键参数
```
```
PGDATABASE=gp_sydb
declare -a DATA_DIRECTORY=(/gpdata/primary)  //节点实例目录
MASTER_HOSTNAME=master    //master安装的主机
MASTER_DIRECTORY=/gpmaster   //master实例安装目录
DATABASE_NAME=test       //master创建的数据库名称
MACHINE_LIST_FILE=/home/gpadmin/gpinitsystem_config //指向配置文件路径
(如果配置镜像，需根据实际情况配置declare -a MIRROR_DATA_DIRECTORY=(/data/mirror)等参数)
```
5) 环境变量配置
在主节点进行环境变量配置
```
vi /home/gpadmin/.bashrc 在最后添加
 
source /usr/local/greenplum-db/greenplum_path.sh
export MASTER_DATA_DIRECTORY=/data/master/gpseg-1
export GPPORT=5432
export PGDATABASE=gp_sydb
```
然后依次复制到各个子节点
```
scp /home/gpadmin/.bashrc gpseg1:/home/gpadmin/
# 让环境变量生效去到对应的节点下
source .bashrc
```
6) 实例初始化（master主机gpadmin执行）

```
su gpadmin
source /usr/local/greenplum-db/greenplum_path.sh
gpinitsystem -c /home/gpadmin/gpinitsystem_config -h /home/gpadmin/seg_hosts_file
```

7) 初步使用(master主机gpadmin执行）
```
gpstate    //检查Greenplum状态
gpstop -r 停止所有实例，然后重启系统
gpstop -u 重新加载配置文件 postgresql.conf 和 pg_hba.conf
查看segment配置：select * from gp_segment_configuration
```