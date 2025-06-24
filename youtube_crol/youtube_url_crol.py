from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os

# 저장된 URL 파일 경로
URL_FILE = "video_urls.txt"

# 이미 저장된 URL 읽기
def load_saved_urls():
    if os.path.exists(URL_FILE):
        with open(URL_FILE, "r", encoding="utf-8") as file:
            return set(file.read().splitlines())  # 중복 방지를 위해 set 사용
    return set()

# URL 저장하기
def save_urls(urls):
    with open(URL_FILE, "a", encoding="utf-8") as file:
        for url in urls:
            file.write(url + "\n")

# YouTube URL 크롤러
def youtube_url_crawler(url):
    service = Service("C:/education/tools/chromedriver-win64/chromedriver.exe")  # chromedriver 경로를 지정
    driver = webdriver.Chrome(service=service)

    try:
        # YouTube 동영상 URL
        driver.get(url)
        driver.maximize_window()

        # 페이지 로드 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "above-the-fold"))
        )
        print("Url 페이지 로드 완료")

        # 스크롤을 내려 동영상 URL 로드
        body = driver.find_element(By.TAG_NAME, 'body')
        for _ in range(30):  # 스크롤을 내리는 횟수 조정
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1)
        
        # 페이지 소스 가져오기
        html_source = driver.page_source
        soup = BeautifulSoup(html_source, 'html.parser')

        # 이미 저장된 URL 불러오기
        saved_urls = load_saved_urls()
        new_urls = []

        # 동영상 URL 가져오기
        for element in soup.find_all('a', {'id': "video-title-link"}):
            href = element.get('href')
            if href:
                full_url = f"https://www.youtube.com{href}"
                if full_url not in saved_urls:  # 중복되지 않은 URL만 추가
                    new_urls.append(full_url)
                    yield full_url  # 제너레이터로 반환

        # 새 URL 저장
        save_urls(new_urls)

    except Exception as e:
        print(f"Url 크롤링 중 오류 발생: {e}")
    finally:
        # WebDriver 종료
        driver.quit()