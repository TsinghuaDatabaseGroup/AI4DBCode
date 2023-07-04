# CentOS7 Installation of Greenplum

Installation environment: CentOS 7.5
Installation version: Greenplum-6.21.0
Installation machines: 182.92.64.198 (host) 39.105.229.60 (node 1)

### root user, disable firewall
```
chkconfig iptables off
systemctl stop firewalld.service
systemctl disable firewalld.service
```
![](media/16593430874831.jpg)

### root user, disable SELINUX
```
vim /etc/selinux/config
```

```
# SELINUX=disabled
```
![](media/16593431072046.jpg)

### root user, modify hosts file (perform on each node)
```
vim /etc/hosts
```
```
182.92.64.198    gpmaster
39.105.229.60   gpseg1
```
![](media/16593433130201.jpg)

### root user, create group and user (perform on each node)
```
groupadd -g 530 gpadmin
useradd -g 530 -u530 -m -d /home/gpadmin -s /bin/bash gpadmin
chown -R gpadmin:gpadmin /home/gpadmin/
passwd gpadmin
```

### Install dependencies
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

Copy to other nodes in sequence
```
scp /etc/sysctl.conf gpseg1:/etc
scp /etc/security/limits.conf gpseg1:/etc/security
scp /etc/selinux/config gpseg1:/etc/selinux
```

Finally, make the node configuration effective by executing the following command on each node
```
sysctl -p
```

### root user, upload the package and elevate file permissions
Package download link: https://github.com/greenplum-db/gpdb/releases/download/6.21.0/open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm
Upload or download to: `/home/gpadmin` directory
```
scp -r open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm root@39.105.229.60:/home/gpadmin/
```
Using sudo (or as root), use the system's package manager to install Greenplum Database package on all hosts
```
cd /home/gpadmin/
chmod +x open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm 
rpm -ivh open-source-greenplum-db-6.21.0-rhel7-x86_64.rpm 
# Installed to /usr/local by default, grant permissions to gpadmin
chown -R gpadmin:gpadmin /usr/local
# Set environment variables
source /usr/local/greenplum-db/greenplum_path.sh
```

### Install GP software on segment server (gpseg1)
1) Establish trust
```
vi /home/gpadmin/all_hosts_file
```
```
gpmaster
gpseg1
```
Establish gpadmin trust: (gpadmin operation)
```
su gpadmin
```
```
ssh-keygen
ssh

-copy-id gpseg1
source /usr/local/greenplum-db/greenplum_path.sh 
gpssh-exkeys -f /home/gpadmin/all_hosts_file  
```

2) Create users, directories, and folders for segment servers

First, create a segment host inventory file (gpadmin)
```
vi /home/gpadmin/seg_hosts_file
```
```
gpseg1
```

3) Preparation before instance initialization
Synchronize system clocks before initialization (gpadmin)

```
1) Check if the system clocks of each segment server are synchronized

$ gpssh -f /home/gpadmin/all_hosts_file -v date

2) Synchronize the clocks of all servers

$ gpssh -f /home/gpadmin/all_hosts_file -v ntpd
```

4) Modify Greenplum database configuration file
```
cp /usr/local/greenplum-db-6.21.0/docs/cli_help/gpconfigs/gpinitsystem_config /home/gpadmin/gpinitsystem_config  //Copy the template configuration file
```
```
vi  /home/gpadmin/gpinitsystem_config  //Modify key parameters
```
```
PGDATABASE=gp_sydb
declare -a DATA_DIRECTORY=(/gpdata/primary)  //Node instance directory
MASTER_HOSTNAME=master    //Host where master is installed
MASTER_DIRECTORY=/gpmaster   //Master instance installation directory
DATABASE_NAME=test       //Name of the database created by master
MACHINE_LIST_FILE=/home/gpadmin/gpinitsystem_config //Path to the configuration file
(If mirroring is configured, configure parameters such as declare -a MIRROR_DATA_DIRECTORY=(/data/mirror) according to the actual situation)
```
5) Environment variable configuration
Configure environment variables on the master node
```
vi /home/gpadmin/.bashrc Add at the end
 
source /usr/local/greenplum-db/greenplum_path.sh
export MASTER_DATA_DIRECTORY=/data/master/gpseg-1
export GPPORT=5432
export PGDATABASE=gp_sydb
```
Then copy to each child node in sequence
```
scp /home/gpadmin/.bashrc gpseg1:/home/gpadmin/
# Make the environment variables take effect and go to the corresponding node
source .bashrc
```
6) Instance initialization (execute by gpadmin on the master host)

```
su gpadmin
source /usr/local/greenplum-db/greenplum_path.sh
gpinitsystem -c /home/gpadmin/gpinitsystem_config -h /home/gpadmin/seg_hosts_file
```

7) Initial usage (execute by gpadmin on the master host)
```
gpstate    //Check Greenplum status
gpstop -r //Stop all instances, then restart the system
gpstop -u //Reload configuration files postgresql.conf and pg_hba.conf
select * from gp_segment_configuration // Check segment configuration
```