#!/usr/bin/env bash

 install_dep(){
     if ! rpm -qa | grep git-lfs; then
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
        sudo yum install -y git-lfs
        git lfs install
      fi

     pkg_list=( java-1.8.0-openjdk-devel git cmake patch libtool automake yasm zlib-devel centos-release-scl devtoolset-7-gcc-c++ )
     for pkg in ${pkg_list[*]}; do
         if ! rpm -qa | grep ${pkg}; then
            sudo yum install -y $pkg
         fi
     done;
 }

echo "Checking dependencies"
install_dep
echo "Change"
echo "Setting GCC"
source scl_source enable devtoolset-7
gcc --version

echo "Setting proxy"
PROXY_HOST=`echo $https_proxy | tr -s '/' ':' | cut -d ':' -f2`
PROXY_PORT=`echo $https_proxy | tr -s '/' ':' | cut -d ':' -f3`
export JAVA_PROXY="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT"

echo "Clean and Build"
./gradlew $JAVA_PROXY clean
./gradlew $JAVA_PROXY build
