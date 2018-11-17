docker run --rm -ti -v `pwd`/../rpm:/home/rpm centos:7.5.1804 bash -c 'yum install -y createrepo; createrepo --database /home/rpm/centos7'
