from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver=webdriver.Chrome()

driver.get('https://pixabay.com/ko/')

elem=driver.find_element_by_name('q')
elem.send_keys('마스크 얼굴')
elem.send_keys(Keys.RETURN)

scroll_pause_time=1
count=1

images=driver.find_elements_by_css_selector('.item')

for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl=driver.find_element_by_css_selector('.item').get_attribute('src')
        urllib.request.urlretrieve(imgUrl, str(count)+'.jpg')
        count=count+1
    except:
        pass

driver.close()