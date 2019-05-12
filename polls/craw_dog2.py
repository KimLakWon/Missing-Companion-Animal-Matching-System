# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from konlpy.tag import Kkma
from polls.dbcon import DBConnect
from polls.predict2_c import recog
from polls.predict_c import breeds
from polls.color_classification_image_c import color
from polls.pre_c import dog


def craw():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(executable_path='/home/ec2-user/py/missing_real/polls/chromedriver', chrome_options=chrome_options)
    driver.implicitly_wait(3)
    db = DBConnect()

    last_num = db.Max() + 1
    final_num = last_num
    cnt = 0
    while True:
        url = 'http://www.zooseyo.or.kr/Yu_board/petfind_view_skin_1.html?no=' + str(final_num)
        driver.get(url)
        element = driver.find_element_by_xpath('//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/span/b').text
        element = element.replace(" ", "")
        if element[4] == '-':
            final_num = final_num + 1
            cnt = cnt + 1
        else:
            cnt = 0
            final_num = final_num + 1
        if cnt == 10:
            break

    final_num = final_num - 11

    while True:
        driver.get('http://www.zooseyo.or.kr/Yu_board/petfind.html')
        driver.find_element_by_xpath('/html/body/table/tbody/tr/td/table/tbody/tr[2]/td[2]/table[1]/tbody/tr/td[2]/table/tbody/tr/td/table[5]/tbody/tr/td[2]/table/tbody/tr[1]/td[1]/table/tbody/tr[1]/td/table/tbody/tr/td/p/a').click()
        url = driver.current_url
        while True:
            if final_num >= last_num:
                #last_num = 23558
                URL = 'http://www.zooseyo.or.kr/Yu_board/petfind_view_skin_1.html?no=' + str(last_num)
                last_num = last_num+1
                driver.get(URL)
                print("URL: " + URL)
                UNO = 0
                IsComplete = 0
                Ptype = 0
                Phone = 0
                # Phone
                element = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/span/b').text
                element = element.replace(" ", "")
                if element[0] == '☆':
                    IsComplete = 1
                    continue
                else:
                    Phone = element[4:]
                    print("Phone: " + Phone)
                    if Phone[0] == "-":
                        continue
                # imgUrl
                Img = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/img[@src]')
                imgUrl = Img.get_attribute('src')
                print("imgUrl: " + imgUrl)
                sys.path.remove(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
                Type = recog(imgUrl)
                print("Type: " + Type)
                if Type == 'cat':
                    Breed = breeds(imgUrl)
                else:
                    Breed = dog(imgUrl)
                if Breed is None:
                    Breed = "None"

                print("Breed : " + Breed)
                #Color
                Color = color(imgUrl)
                print("Color : " + Color)
                #asd
                path = driver.find_elements_by_xpath('//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]|//p')

                if len(path) == 3:
                    Reward_text = driver.find_element_by_xpath(
                        '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/p[1]/font/span/b').text

                    Reward = Reward_text[6:]
                else:
                    Reward = "협의"

                print("Reward: " + Reward)
                # LostWhere
                LostWhere = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[7]/td/table/tbody/tr/td[2]/table/tbody/tr[1]/td/table/tbody/tr/td[2]/b').text
                print("LostWhere: " + LostWhere)
                # LostWhen
                LostWhen = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[7]/td/table/tbody/tr/td[2]/table/tbody/tr[1]/td/table/tbody/tr/td[4]/b').text
                print("LostWhen: " + LostWhen)
                # Gender
                Gen = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[7]/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td/table/tbody/tr/td[2]/b[3]').text
                print("Gender: " + Gen)
                if Gen == '수컷':
                    Gender = 'M'
                else:
                    Gender = 'F'
                # Contents
                Contents = driver.find_element_by_xpath(
                    '//*[@id="printTable"]/table/tbody/tr/td/table/tbody/tr[1]/td[2]/table/tbody/tr[2]/td/table/tbody/tr[7]/td/table/tbody/tr/td[2]/table/tbody/tr[5]/td/table/tbody/tr/td[2]/b').text
                Contents = Contents.replace("\n", " ")
                print("Contents: \n" + Contents)
                kkma = Kkma()

                Contents_konlpy = kkma.nouns(Contents)

                print("Contents_konlpy : ")
                print(Contents_konlpy)

                db = DBConnect()
                db.make(UNO, Ptype, imgUrl, LostWhere, LostWhen, Phone, Gender, Type, Breed, Color, Reward, Contents, Contents_konlpy, IsComplete, URL)
            else:
                break

        return

