#import 
import requests
from bs4 import BeautifulSoup

#data in
site = requests.get("https://paiza.cloud/ja/")
data = BeautifulSoup(site.text, "html.parser")

#ページのタイトルを出力
print(data.title.text)

#aタグの内容を表示
print(data.find_all('a')) 

#id属性。id_nameに一位するタイトル
print(data.find(id = 'id_name'))

#特定ワード「Google」に完全に一致している文字列を出力
print(data.find(text = 'Google'))
