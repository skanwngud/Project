from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

# select webbrowser (chrome)
driver=webdriver.Chrome()

# link to address
driver.get('https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl')

# fine specified elements
elem=driver.find_element_by_name('q')

# input keys & enter
elem.send_keys('shtterstock mask face')
elem.send_keys(Keys.RETURN)

# scroll web page
SCROLL_PAUSE_TIME=1
last_height=driver.execute_script('return document.body.scrollHeight')
# 스크롤 높이를 java Script 로 찾아서 last_height 란 변수에 저장 시킴

while True: # 무한 반복
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    # 스크롤을 끝까지 내린다

    time.sleep(SCROLL_PAUSE_TIME) # 스크롤이 끝나면 1초동안 기다림

    new_height=driver.execute_script('return document.body.scrollHeight')
    if new_height==last_height:
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
            # 결과 더보기 버튼 클릭
        except:
            break
    last_height=new_height

# select & click image in webbrowser 
images=driver.find_elements_by_css_selector('.rg_i.Q4LuWd')
count=1

for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl=driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')
        # opener=urllib.request.build_opener()
        # opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        # urllib.request.install_opener(opener)
        # urllib.request.urlretrieve(imgUrl, "test.jpg")
        # Forbidden 이 뜨면 위의 코드를 추가한다
        time.sleep(2)
        urllib.request.urlretrieve(imgUrl, 'shutterstock_mask'+str(count)+'.jpg')
        count=count+1
    except:
        pass
driver.close()

