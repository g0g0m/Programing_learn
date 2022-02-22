#import 
import requests
from bs4 import BeautifulSoup

#data in
site = requests.get("")
data = BeautifulSoup(site.text, "html.parser")

print(data.find_all('a class'))