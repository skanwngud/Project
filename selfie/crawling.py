from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

# 1. 웹브라우저와 드라이버를 선택 (크롬)
# chromedriver.exe 파일이 crawling.py 파일과 같은 경로에 있어야함
driver = webdriver.Chrome()

# 2. 크롬 브라우저를 이용해 해당 주소로 이동함
driver.get('https://pixabay.com/ko/photos/search//?cat=people')

# 3. 웹페이지의 특정 요소를 찾음 (검색을 하기 위함)
elem = driver.find_element_by_name('q')

# 4. 원하는 검색어를 입력
elem.send_keys('초상화 사람 얼굴')

# 5. enter 키 입력
elem.send_keys(Keys.RETURN)

# 6. 사진의 각 요소를 선택 및 클릭
driver.find_elements_by_css_selector('.photo-result-image')[0].click()
time.sleep(3)
imgUrl = driver.find_elements_by_css_selector('.media_container').get_attribute('src')
urllib.request.urlretrieve(imgUrl, 'test.jpg')