# 0. 라이브러리 임포트
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

# 1. 웹브라우저 및 드라이버 선택
driver=webdriver.Chrome() # Firefox, IE 등도 존재

# 2. Chrome 브라우저를 이용해 해당 주소로 들어감
driver.get('https://www.google.co.kr/imghp?hl=ko&ogbl')

# 3. 웹페이지의 특정 요소를 찾음
elem=driver.find_element_by_name('q')
# find_element, element_by_class, id ... 등

# 4. 원하는 값(검색어)을 입력
elem.send_keys('마스크 쓴 얼굴 사진')

# 5. enter 키 전송
elem.send_keys(Keys.RETURN) 
# 해당 코드를 모르면 구글에 검색

# 6. 사진의 각 요소를 선택 및 클릭
driver.find_elements_by_css_selector('.rg_i.Q4LuWd')[0].click()
time.sleep(3) # 사진을 불러오는데에 어느 정도의 시간이 필요하기 때문에 해당 코드를 넣어 3초간 고의로 지연시킨다
# elements 를 봤을 때 여러개의 선택 된 이미지들 중 가장 첫 번째 [0] 의 것만 클릭
# driver.find_elements_by_css_selector('.rg_i.Q4Luwd').click()
# 선택 된 이미지들을 전부 클릭
# class 면 연결부에 . 을 붙인다
# element, elements 의 차이점은 전자는 하나의 요소, 후자는 복수의 요소를 갖는다
imgUrl=driver.find_element_by_css_selector('.n3VNCb').get_attribute('src')
urllib.request.urlretrieve(imgUrl, 'test.jpg') # 해당 주소의 이미지를 저장
# 크롬에서는 사진을 다운 받기 위해선 한 번 클릭한 뒤 큰 이미지로 나온 뒤
# 다시 다운로드를 진행해야하기 때문에 큰 사진이 나오고 그거에 대한 이미지 주소를 불러온다


# assert 'Python' in driver.title
# elem.clear()
# assert 'No results found.' not in driver.page_source
# driver.close()