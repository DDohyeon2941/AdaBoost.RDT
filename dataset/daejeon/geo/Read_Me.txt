1. 작성일: 2023-03-16
2. 목적: 지리적 환경변수를 추출하는 과정에 대한 전반적인 내용을 설명하고자 함
3. 데이터 다운로드

1) 대전 시내버스 정류소
https://www.data.go.kr/data/15081730/fileData.do 
에서 다운받음
	- 해당 데이터셋의 경우, 위경도가 도와 분으로 나타나 있음 >> 이를 도와분을 좌표값으로 바꿔서 거리를 계산함. 이때 거리는 haversine, unit은 m

2) 자전거 대여소 정보
나스 서버내, 정주현 학생이 업로드한 정보를 활용함
ftp://dmfile.ipdisk.co.kr:2222/HDD2/Data/Project/%BF%AC%B1%B8%C0%E7%B4%DC/Bike%20Sharing/Data/1_Bike%20Sharing/Daejeon%20Bike/


3) 대전 대학교 정보
https://www.daejeon.go.kr/drh/DrhContentsHtmlView.do?menuSeq=1663 에서 다운받음

4) kakao api
'63e2b38e5cd98fb09aeba2634f72d6db'

4. 주요 프로세스

1) 대전 대학교 주소를 기반해 좌표값 구하기
	- 정류소별 가장 가까이 위치한 대학교와의 거리 산출
2) 대여소별 다른 대여소와의 거리 구하기 >> 특정 반경내의 자전거 정류소 수 구하기
	- 소스파일에는 300여개가 있으나, 실제로 활용하는 정류소는 262곳임
	- 특정 반경내 위치하더라도, 실제로 활용하는 정류소가 아니면, 카운트 하지 않음

3) 카테고리별 정보 수집
	카테고리: ('CE7', '카페'), ('FD6', '음식점'), ('BK9', '은행'), ('SW8', '지하철'), ('AT4', '관광')
	코드는 아래와 같음
    
def get_cate_size_uni(lon, lat, radius, cate_code, cate_query, url, headers):
    params = {'query' : cate_query, 'x' : lon, 'y' : lat, 'radius' : radius, 'category_group_code' : cate_code}
    total = requests.get(url, params=params, headers=headers).json()
    return total['meta']['total_count']