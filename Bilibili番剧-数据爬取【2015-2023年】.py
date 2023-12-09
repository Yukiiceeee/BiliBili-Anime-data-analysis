import requests
import json
import pandas as pd

score_list=[]
ep_list=[]
#设置爬取页数
page=11
headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
url = 'https://api.bilibili.com/pgc/season/index/result'
for i in range(page):
        param = {
            'st': '1',
            'order': '3',
            'season_version': '-1',
            'spoken_language_type': '-1',
            'area': '-1',
            'is_finish': '-1',
            'copyright': '-1',
            'season_status': '-1',
            'season_month': '-1',
            # 可修改year值
            'year': '[2017,2018)',
            'style_id': '-1',
            'sort': '0',
            'page': i,
            'season_type': '1',
            'pagesize': '20',
            'type': '1',
        }
        try:
                response = requests.get(url=url, params=param, headers=headers)
                data = json.loads(response.text)
                json_list = data['data']['list']
                # 获取ep_id，score
                for i in range(len(json_list)):
                    score = json_list[i]['score']
                    ep_id=json_list[i]['first_ep']['ep_id']
                    ep_list.append(ep_id)
                    score_list.append(score)
        except:
            print('未爬取到数据')

# 存储所有数据
ff=[]
for i in range(len(ep_list)):
    ep_id=ep_list[i]
    # title=title_list[i]
    score=score_list[i]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    url2='https://api.bilibili.com/pgc/view/web/season?ep_id='+str(ep_id)

    response = requests.get(url=url2, headers=headers)
    data = json.loads(response.text)

    json_list = data['result']

    title = json_list['title']
    views = json_list['stat']['views']
    likes = json_list['stat']['likes']
    coins = json_list['stat']['coins']
    shares = json_list['stat']['share']
    danmakus = json_list['stat']['danmakus']
    series_follow = json_list['stat']['follow_text']
    styles=json_list['styles']

    dic = {
        '名称': title,
        '播放量': views,
        '评分': score,
        '点赞量': likes,
        '投币量': coins,
        '转发量': shares,
        '弹幕数': danmakus,
        '系列追番数': series_follow,
        '所属类别':styles
    }

    ff.append(dic)
    print(ff)
    # 利用pd导出表格
    pd_data = pd.DataFrame(ff)
    pd_data.to_excel('2017年B站番剧数据.xlsx')

