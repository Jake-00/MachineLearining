# 参考博客：https://www.cnblogs.com/rednodel/p/4565539.html
import os

if __name__ == "__main__":
    read_file = "D:\chrome_download\Python3_MachineLearing\datingTestSet2.txt"
    write_file = os.path.splitext(read_file)[0] + '_reshape.txt'
    with open(read_file) \
            as file_dir:
        with open(write_file, 'w', encoding='utf-8') \
                as file_process:
            lines_ = file_dir.readlines()
            for line in lines_:
                line = line.replace('\t', ',')  # 去除字符串换行符
                file_process.write(line)


