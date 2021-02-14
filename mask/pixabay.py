from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver=webdriver.Chrome()

driver.get('https://www.shutterstock.com/ko/search?image_type=photo')

elem=driver.find_element_by_css_selector('.o_input_theme_input-element.oc_Y_4ccd2.b_J_d43c3.o_b_608c4')
# elem=driver.find_element_by_css_selector('.searchterm')
elem.send_keys('사람 얼굴')
elem.send_keys(Keys.RETURN)

scroll_pause_time=1
count=1
time.sleep(2)

# images=driver.find_elements_by_css_selector('.z_h_9d80b.z_h_2f2f0')
images=driver.find_elements_by_xpath('/html/body/div[2]/div[2]/div/div[2]/div/main/div/div[2]/div/div[2]/div/a/img')


# for image in images:
#     try:
#         image.click()
#         time.sleep(2)
#         imgUrl=driver.find_element_by_css_selector('.m_l_c4504').get_attribute('src')
#         urllib.request.urlretrieve(imgUrl, 'shutter_face' + str(count) + '.jpg')
#         driver.execute_script("window.history.go(-1)")
#         time.sleep(2)
#         count+=1
#     except:
#         pass

# for image in images:
#     try:
#         image.click()
#         time.sleep(2)
#         imgUrl=driver.find_element_by_css_selector('.m_l_c4504').get_attribute('src')
#         urllib.request.urlretrieve(imgUrl, str(count)+'.jpg')
#         count=count+1
#     except:
#         pass

# driver.close()