import pymysql
from datetime import datetime
# 클래스 선언
class DBConnect:
    # 생성자 선언
    def __init__(self):
        # MySQL DB와 연결을 위한 connect 인스턴스를 선언
        self.conn = pymysql.connect(host='localhost', port=3306, user='root', password='rootpswd', db='missing', charset='utf8')
        # 파이썬에서 쿼리를 사용하고 저장하기 위한 cursor 인스턴스 선언
        self.curs = self.conn.cursor(pymysql.cursors.DictCursor)
    # db 자료 분류
    def make(self, UNO, Ptype, imgUrl, LostWhere, LostWhen, Phone, Gender, Type, Breed, Color, Reward, Contents, Contents_konlpy, IsComplete, URL):
        self.insert(UNO, Ptype, imgUrl, LostWhere, LostWhen, Phone, Gender, Type, Breed, Color, Reward, Contents, Contents_konlpy, IsComplete, URL)
    # 간단한 insert 메서드 정의
    def insert(self, UNO, Ptype, imgUrl, LostWhere, LostWhen, Phone, Gender, Type, Breed, Color, Reward, Contents, Contents_konlpy, IsComplete, URL):
        try:
            # insert 쿼리 작성
            qry = "SELECT MAX(PNO) FROM poster"
            self.curs.execute(qry)
            result = self.curs.fetchone()
            print(result['MAX(PNO)'])
            if str(result['MAX(PNO)']) == 'None':
                PNO = 1
            else:
                PNO = int(result['MAX(PNO)']) + 1

            konlpy = ""
            for item in Contents_konlpy:
                konlpy += item
                konlpy += " "

            sql = "INSERT INTO poster VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (PNO, UNO, Ptype, imgUrl, LostWhere, LostWhen, Phone, Gender, Type, Breed, Color, Reward, Contents, konlpy, IsComplete, URL)
            print(val)
            self.curs.execute(sql, val)
            # DML문 완료 후 커밋
            self.conn.commit()
        except Exception as e:
            print(e)
        finally:
        # 커넥트 인스턴스를 닫아준다.
            self.conn.close()
    def Max(self):
        try:
            # insert 쿼리 작성
            qry = "SELECT MAX(URL) FROM poster"
            self.curs.execute(qry)
            result = self.curs.fetchone()
            print(result['MAX(URL)'])
            max = str(result['MAX(URL)'])
            n = max.find('=') + 1
            print(max[n:])
            max_no = int(max[n:])
            return max_no
        except Exception as e:
            print(e)
        finally:
        # 커넥트 인스턴스를 닫아준다.
            self.conn.close()
