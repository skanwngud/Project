from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import time
import urllib.request

class WebDriver:
    def __init__(self):
        self.__driver = webdriver.Chrome()
        self.__element = ""
        
    def get_address(self, address):
        self.__driver.get(address)
        
    def search(self, tag, search_value):
        self.__element = self.__driver.find_element_by_name(tag)
        self.__element.send_keys(search_value)
        self.__element.send_keys(Keys.RETURN)
        
        
        
if __name__ == "__main__":
    run = WebDriver()
    run.get_address("https://www.google.co.kr/imghp?=hlko&ogbl")
    run.search('q', 'test')