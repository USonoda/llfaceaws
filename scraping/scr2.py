import os
import sys
from selenium import webdriver
from bs4 import BeautifulSoup
from time import time, sleep
from urllib.request import urlopen, Request
from urllib.parse import quote
from mimetypes import guess_extension
import traceback
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Fetcher:
    def __init__(self, ua=''):
        self.ua = ua

    def fetch(self, url):
        req = Request(url, headers={'User_Agent': self.ua})
        try:
            with urlopen(req, timeout=3) as p:
                b_content = p.read()
                mime = p.getheader('Content-Type')
        except:
            sys.stderr.write('Error in fetching {}\n'.format(url))
            sys.stderr.write(traceback.format_exc())
            return None, None
        return b_content, mime


fetcher = Fetcher('Mozilla/5.0')  # ユーザーエージェント
num = 1
dirname = 'data'


# ディレクトリ作成、ｎページ検索、保存をする関数
def scrape(word, page_num, dir_name):  # 検索ワード、保存するページ数、保存先
    # 画像保存用ディレクトリ作成
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # PhantomJSをSelenium経由で利用
    url = 'http://image.search.yahoo.co.jp/search?n=60&p={}&search.x=1'.format(quote(word))
    driver = webdriver.PhantomJS()

    # PhantomJSで該当ページを取得&レンダリング
    driver.get(url)

    for i in range(page_num):
        # レンダリング結果をPhantomJSから取得
        html = driver.page_source

        # 画像のurlを取得する
        bs = BeautifulSoup(html, "html.parser")
        img_urls = [img.get("href") for img in bs.find_all("a", target="imagewin")]  # urlリストの作成
        img_urls.remove("javascript:void(0);")
        img_urls = list(set(img_urls))  # 重複を除く
        # 画像を保存する
        for j, img_url in enumerate(img_urls):
            sleep(0.1)
            img, mime = fetcher.fetch(img_url)
            if not mime or not img:
                continue
            ext = guess_extension(mime.split(';')[0])
            if ext in ('.jpe', '.jpeg'):  # 拡張子変更
                ext = '.jpg'
            if not ext:
                continue
            result_file = os.path.join(dir_name, str(i) + '_' + str(j) + ext)  # 名前づけ
            with open(result_file, mode='wb') as f:  # 保存
                f.write(img)
            print('fetched', img_url)

        # 次のページに移動する
        driver.find_element_by_link_text('次へ>').click()


if __name__ == '__main__':
    word = sys.argv[1]
    scrape(word, num, dirname)
