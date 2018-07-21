# -*- coding:utf-8 -*-
# author: zhaozhichao
# use you-get download bilibili video
# sudo pip install you-get
# version 1.0

import os
import time

def download_single_video(video_url, savepath, savename):
    cmd = "you-get --playlist -o " + savepath + " -O " + savename + " " + video_url
    print(cmd)
    os.system(cmd)

def download(savepath, baseurl, start_index=0, end_index=20):
    for i in range(start_index, end_index+1, 1):
        video_url  = baseurl + str(i)    
        print("download {0:} video start".format(i))
        download_single_video(video_url, savepath, "video_" + str(i))
        print("download {0:} video end".format(i))
        time.sleep(2)

# add error handle , skip it

if __name__ == '__main__':
    savepath = "./video"
    if not os.path.exists(savepath):
        os.makedirs(savepath)  
    baseurl = "https://www.bilibili.com/video/av25569107/?p="
    start_index = 0
    end_index = 3
    download(savepath, baseurl, start_index, end_index)


# TODO:
# 1. config file
# 2. error check
# 3. user interface
# 4. use more you_get function






