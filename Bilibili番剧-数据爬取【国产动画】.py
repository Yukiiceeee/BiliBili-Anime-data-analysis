import requests
import json
import pandas as pd

score_list = []
ep_list = []

# 设置爬取页数
page = 10
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
}
url = 'https://api.bilibili.com/pgc/season/index/result'

for i in range(page):
    param = {
        'season_version': '-1',
        'is_finish': '-1',
        'copyright': '-1',
        'season_status': '-1',
        'year': '[2015,2016)',
        'style_id': '-1',
        'order': '3',
        'st': '4',
        'sort': '0',
        'page': i,
        'season_type': '4',
        'pagesize': '20',
        'type': '1',
    }
    try:
        response = requests.get(url=url, params=param, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        data = json.loads(response.text)
        json_list = data['data']['list']

        # 获取ep_id，score
        for j in range(len(json_list)):
            score = json_list[j]['score']
            ep_id = json_list[j]['first_ep']['ep_id']
            ep_list.append(ep_id)
            score_list.append(score)
    except requests.exceptions.RequestException as e:
        print(f'请求失败：{e}')

# 存储数据
ff = []
for k in range(len(ep_list)):
    ep_id = ep_list[k]
    score = score_list[k]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
    }
    url2 = 'https://api.bilibili.com/pgc/view/web/season?ep_id=' + str(ep_id)

    try:
        response = requests.get(url=url2, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        data = json.loads(response.text)

        json_list = data['result']

        title = json_list['title']
        views = json_list['stat']['views']
        likes = json_list['stat']['likes']
        coins = json_list['stat']['coins']
        shares = json_list['stat']['share']
        danmakus = json_list['stat']['danmakus']
        series_follow = json_list['stat']['follow_text']
        styles = json_list['styles']
    except requests.exceptions.RequestException as e:
        print(f'请求失败：{e}')


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
    pd_data = pd.DataFrame(ff)
    pd_data.to_excel('2015年国产B站番剧数据.xlsx')