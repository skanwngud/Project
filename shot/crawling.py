from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver = webdriver.Chrome()
driver.get("https://golfcritic.co.kr/php/page/ground_list.php?country_idx=1&cnt_ground_idx=15171&cnt_scroll_value=68")

