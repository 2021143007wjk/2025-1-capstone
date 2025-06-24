from youtube_vidio_crol import youtube_comments_crawler
from youtube_url_crol import load_saved_urls, save_urls, youtube_url_crawler

saved_urls = load_saved_urls()

for video_url in youtube_url_crawler('https://www.youtube.com/@BBCNews/videos'):
    if video_url in saved_urls:
        print(f"이미 처리된 URL: {video_url}")
        continue

    try:
        print(f"크롤링 시작: {video_url}")
        data = youtube_comments_crawler(video_url)
        data.to_csv(f'crol_data/youtube_comments_{video_url.split('=')[1]}.csv', index=False, encoding='utf-8-sig')
        print(f"댓글 데이터를 {f'crol_data/youtube_comments_{video_url.split('=')[1]}.csv'} 파일로 저장했습니다.")

        # URL 저장
        save_urls([video_url])

    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")