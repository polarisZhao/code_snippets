# -*- coding:utf-8 -*-
# ////////////////////////////////////////////////////////////////
#
#  Download Image from URL
#  Authors: zhaozhichao
#
# ////////////////////////////////////////////////////////////////
import requests
from contextlib import closing

def download_image(url):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept':'text/html;q=0.9,*/*;q=0.8',
    'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding':'gzip',
    'Connection':'close',
    'Referer':None
    }
    # url = "http://img.ph.126.net/ocT0cPlMSiTs2BgbZ8bHFw==/631348372762626203.jpg"
    with closing(requests.get(url, headers = headers, stream=True)) as response:
        # print(response.content)
        with open('demo.jpg', 'wb') as fd:
            for chunk in response.iter_content(128):
                fd.write(chunk)

def main():
    url = "http://img.ph.126.net/ocT0cPlMSiTs2BgbZ8bHFw==/631348372762626203.jpg"
    download_image(url)

if __name__ == '__main__':
    main()
