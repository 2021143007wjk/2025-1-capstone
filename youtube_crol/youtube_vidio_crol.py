from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# 댓글 크롤링
def main_reply(find_soup, main_reply_list, i):
    span_element = find_soup[i].find('span', {'class': "yt-core-attributed-string yt-core-attributed-string--white-space-pre-wrap", 'dir': "auto"})
    if span_element and span_element.text:
        text = span_element.text.strip()
    else:
        text = "댓글 없음"
        pass
    main_reply_list.append(text)

# 대댓글 갯수 크롤링
def reply_reply(find_soup, reply_reply_list, i):
    replies_div = find_soup[i].find('div', {'id': "replies"})
    if replies_div and replies_div.text:
        reply = replies_div.text.strip().split('\n')[0]
    else:
        reply = "답글 0개"
    reply_reply_list.append(reply)

#댓글의 좋아요 갯수 크롤링
def reply_good(find_soup, reply_good_list, i):
    good_div = find_soup[i].find('span', {'id':"vote-count-middle"})
    if good_div:
        good = good_div.text.strip()
    else:
        good = "0"
    reply_good_list.append(good)


# Selenium WebDriver 설정
def youtube_comments_crawler(url):
    service = Service("C:/education/tools/chromedriver-win64/chromedriver.exe")  # chromedriver 경로를 지정
    driver = webdriver.Chrome(service=service)


    try:
        # YouTube 동영상 URL
        driver.get(url) 
        driver.maximize_window()

        # 페이지 로드 대기 (ID가 "above-the-fold"인 요소가 로드될 때까지 대기)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "above-the-fold"))
        )
        print("댓글 페이지 로드 완료")
        # 스크롤을 내려 댓글 로드
        body = driver.find_element(By.TAG_NAME, 'body')
        for _ in range(10):  # 스크롤을 내리는 횟수 조정
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1)
            
        # 페이지 소스 가져오기
        html_source = driver.page_source
        soup = BeautifulSoup(html_source, 'html.parser')

        # 제목과 조회수
        title = soup.find('div', {'id': "above-the-fold", 'class': "style-scope ytd-watch-metadata"}).text.strip().split('\n')[0]
        viewer = soup.find('span', {'class': "bold style-scope yt-formatted-string"}).text.strip().split('\n')[0]
        max_good = soup.find('toggle-button-view-model').text.strip()

        # 댓글 크롤링
        main_reply_list = []
        reply_reply_list = []
        reply_good_list = []

        # 댓글 섹션을 먼저 찾음
        find_soup = soup.find_all('ytd-comment-thread-renderer', {'class': "style-scope ytd-item-section-renderer"})

        for i in range(len(find_soup)):
            # 댓글 텍스트
            main_reply(find_soup, main_reply_list, i)

            # 답글 텍스트
            reply_reply(find_soup, reply_reply_list, i)

            # 댓글 좋아요 수
            reply_good(find_soup, reply_good_list, i)

        # 데이터프레임 생성
        reply_df = pd.DataFrame({
            "댓글": main_reply_list,
            "답글": reply_reply_list,
            "좋아요": reply_good_list,
            "제목": title,
            "조회수": viewer,
            "최대 좋아요": max_good
        })

    except Exception as e:
        print(f"댓글 크롤링 중 오류 발생: {e}")
    finally:
        # WebDriver 종료
        driver.quit()
    return reply_df