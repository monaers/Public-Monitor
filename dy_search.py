import requests
import datetime
from collections import defaultdict

# 抖音开放平台API的URL
SEARCH_URL = "https://open.douyin.com/video/search/"
COMMENT_URL = "https://open.douyin.com/video/search/comment/list/"
ACCESS_TOKEN = "act.1d1021d2aee3d41fee2d2add43456badMFZnrhFhfWotu3Ecuiuka27L56lr"
OPENID = "ba253642-0590-40bc-9bdf-9a1334b94059"

def search_videos(keyword):
    params = {
        'open_id': OPENID,
        'keyword': keyword,
        'cursor': 0,
        'count': 10,
    }
    
    # 发送搜索请求
    headers = {
        'access-token': ACCESS_TOKEN
    }
    response = requests.get(SEARCH_URL, params=params, headers=headers)

def get_comments(sec_item_id):
    comments = []
    cursor = 0
    has_more = True
    encoded_sec_item_id = quote(sec_item_id)
    
    while has_more and len(comments) < 100:
        # 定义获取评论请求的参数
        params = {
            'sec_item_id': encoded_sec_item_id,
            'cursor': cursor,
            'count': 50,
        }
        
        # 发送获取评论的请求
        headers = {
            'access-token': ACCESS_TOKEN
        }
        response = requests.get(COMMENT_URL, params=params, headers=headers)
        
    return comments[:100]  # 返回前100条评论

def convert_timestamp_to_utc8(timestamp):
    timestamp = timestamp
    utc_time = datetime.datetime.utcfromtimestamp(timestamp)
    utc8_time = utc_time + datetime.timedelta(hours=8)
    return utc8_time.strftime('%Y-%m-%d %H:%M:%S')

def main():
    keyword = input("请输入要搜索的关键字：")
    videos = search_videos(keyword)
    
    if videos:
        for video in videos:
            video_id = video['item_id']
            nickname = video['nickname']
            sec_item_id = video['sec_item_id']
            url = video['share_url']
            title = video['title']
            comment_count = video['statistics']['comment_count']
            digg_count = video['statistics']['digg_count']
            forward_count = video['statistics']['forward_count']
            download_count = video['statistics']['download_count']
            create_time = convert_timestamp_to_utc8(video['create_time'])
            
            print(f"\n视频ID: {video_id}")
            print(f"作者昵称: {nickname}")
            print(f"特殊加密的视频ID: {sec_item_id}")
            print(f"视频URL: {url}")
            print(f"视频标题: {title}")
            print(f"评论数: {comment_count}")
            print(f"点赞数: {digg_count}")
            print(f"转发数: {forward_count}")
            print(f"下载数: {download_count}")
            print(f"发布时间（UTC+8）: {create_time}")
            
            comments = get_comments(sec_item_id)
            if comments:
                # 对评论按点赞数进行排序
                comments.sort(key=lambda x: x['digg_count'], reverse=True)
                # 按评论者ID分组
                comment_dict = defaultdict(list)
                for comment in comments:
                    comment_dict[comment['comment_user_id']].append(comment)
                
                # 打印每个用户的顶级评论
                for user_id, user_comments in comment_dict.items():
                    user_comments.sort(key=lambda x: x['digg_count'], reverse=True)
                    top_comment = user_comments[0]
                    comment_create_time = convert_timestamp_to_utc8(top_comment['create_time'])
                    
                    print(f"\n评论者ID: {user_id}")
                    print(f"评论内容: {top_comment['content']}")
                    print(f"评论发布时间（UTC+8）: {comment_create_time}")
                    print(f"评论点赞数: {top_comment['digg_count']}")
                    print(f"评论回复数: {top_comment['reply_comment_total']}")
            else:
                print("没有找到评论。")
    else:
        print("没有找到相关视频。")

if __name__ == "__main__":
    main()
