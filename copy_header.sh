#!/bin/bash
#这里我们只是为了解决问题，更深入的做法是应该抽象出来该类问题的共性，然后固化脚本；
#找到头文件路径列表
head_files=`find /private/var/tmp/_bazel_carrothu/6007271e5c9bdcd2ee2b96b71be675d0/execroot/mediapipe/bazel-out/arm64-v8a-opt/bin/mediapipe -name "*.h"`
des_dir="/Users/carrothu/Desktop/copydir"
file_count=0

[ -d $des_dir ] || mkdir -p $des_dir &> /dev/null

#进行拷贝过程
for file in $head_files
do
    source_dir=${file%/*}
    #判断目标目录是否存在，不存在创建
    [ -d $des_dir$source_dir ] || mkdir -p $des_dir$source_dir
    cp $file $des_dir$source_dir
    echo "$file has been copy."
    ((file_count++))
done

#对拷贝结果进行判断
if [ $? != 0 ]; then
    echo "copy files error!!!"
else
    echo "copy files successfully!,total $file_count files."
fi