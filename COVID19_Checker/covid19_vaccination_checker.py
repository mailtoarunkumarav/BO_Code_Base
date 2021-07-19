from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
#chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--disable-gpu")
#chrome_options.add_argument("--no-sandbox") # linux only
chrome_options.add_argument("--headless")
# chrome_options.headless = True # also works

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.barwonhealth.org.au/coronavirus/booking-a-vaccination/vaccination-under-60-years-of-age")

# main_iframe = driver.find_elements_by_tag_name('iframe')[0]
# driver.switch_to.frame(main_iframe)

mainContainer = driver.find_elements_by_xpath("/html/body/div/div/form/div[6]/div[1]/div/div")

print(mainContainer)

# search_bar = driver.find_element_by_name("q")
# search_bar.clear()
# search_bar.send_keys("getting started with python")
# search_bar.send_keys(Keys.RETURN)
print(driver.current_url)
driver.close()