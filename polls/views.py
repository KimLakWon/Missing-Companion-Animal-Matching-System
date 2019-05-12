from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from polls.models import Question, Choice, MissInform
from django.db import connection
from polls.forms import NameForm
from konlpy.tag import Kkma
from polls.predict2 import recog
from polls.predict import breeds
from polls.color_classification_image import color
from polls.pre import dog
from polls.craw_dog2 import craw
import time
# mysql
def selectTest():
	cursor = connection.cursor()
	query_string = "select * from poster Group by Phone"
	cursor.execute(query_string)
	rows = cursor.fetchall()
	posts = []
	for row in rows:
		if row[14] == 0:#완료된 포스터는 제외
			dic = {'PNO':row[0], 'ImgUrl':row[3], 'LostWhere':row[4],  'LostWhen':row[5], 'Phone':row[6], 'Gender':row[7], 'Type':row[8], 'Breed':row[9], 'Color':row[10], 'Reward':row[11], 'Contents':row[12], 'Contents_konlpy':row[13], 'URL':row[15], 'Point':0}
			posts.append(dic)

	return posts

# Post
def get_name(request):
	if request.method == 'POST':
		form = NameForm(request.POST)
		if form.is_valid():
			new_name = form.cleaned_data['name']
			return HttpResponseRedirect('/search/')
	else:
		form = Nameform()
	return render(request, 'miss.html',{'form':form})
def Insert(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
                        file = request.FILES['file']
                        filename = file._name

                        fp = open('%s%s' % ('/home/ec2-user/py/missing_real/polls/', filename), 'wb')
                        for chunk in file.chunks():
                                fp.write(chunk)
                        fp.close()
    MissInform(
            url = filename,
            text = request.POST.get('text'),
            gender = request.POST.get('gender'),
            city = request.POST.get('city'),
        ).save()

    return HttpResponseRedirect('/polls/search') #추가 후 목록보기
def upload(request):

	if request.method == 'POST':
		if 'file' in request.FILES:
			file = request.FILES['file']
			filename = file._name

			fp = open('%s%s' % ('/home/ec2-user/py/missing_real/polls/', filename), 'wb')
			for chunk in file.chunks():
				fp.write(chunk)
			fp.close()
			return HttpResponse('File Uploaded')
	return HttpResponse('Failed to Upload File')

# Create your views here.
def index(request):
	latest_question_list = Question.objects.all().order_by('-pub_date')[:5]
	context = {'latest_question_list': latest_question_list}
	posts = selectTest()
	return render(request, 'polls/index.html',{'posts':posts})

def miss(request):
	miss = []
	miss2 = MissInform.objects.all()
	return render(request, 'polls/miss.html',{'miss':miss, 'miss2':miss2})

def boho(request):
	print("boho_start!")
	craw()
	print("boho_finish!")
	return render(request, 'polls/boho.html')

def search(request):

	return render(request, 'polls/search.html')
	#return HttpResponseRedirect('/polls/results')

def detail(request, question_id):
	question = get_object_or_404(Question, pk=question_id)
	return render(request, 'polls/detail.html', {'question': question})

def vote(request, question_id):
	p = get_object_or_404(Question, pk=question_id)
	try:
		selected_choice = p.choice_set.get(pk=request.POST['choice'])
	except (KeyError, Choice.DoesNotExist):
		return render(request, 'polls/detail.html', {
			'question' : p,
			'error_message' : "You didn't select a choice.",
		})
	else:
		selected_choice.votes += 1
		selected_choice.save()
		return HttpResponseRedirect(reverse('polls:results', args=(p.id,)))

def results(request):
	#missInform의 Image와 Contents 받기
        late = MissInform.objects.all()
        path_text = late[len(late)-1]
        path = "polls/"+path_text.url
        Type = recog(path)
        if Type == 'cat':
                Breed = breeds(path)
        else:
                Breed = dog(path)
        if Breed is None:
                Breed = "None"
        print(Breed)
        Color = color(path)
        kkma = Kkma()
        text = kkma.nouns(path_text.text)
        posts = selectTest()
        point = []
        for post in posts:
                if post['Type'] == Type:
                        post['Point'] += 100
                if post['Breed'] == Breed:
                        post['Point'] += 70
                if post['Color'] == Color:
                        post['Point'] += 10
                if post['Gender'] == path_text.gender:
                        post['Point'] += 30
                if post['Gender'] == 'O':
                        post['Point'] += 10
                if post['LostWhere'][:2] == path_text.city:
                        post['Point'] += 30
                text2_text = post['Contents_konlpy']
                text2 = text2_text.split()
                cnt = 0
                for i in text:
                        if i in text2:
                                if len(i) < 2:
                                        cnt += 3
                                elif len(i) < 3:
                                        cnt += 10
                                else:
                                        cnt += 20
                if cnt >= 40:
                        post['Point'] += 40
                else:
                        post['Point'] += cnt
        posts = sorted(posts, key=lambda k: k['Point'])
        point = posts[len(posts)-10:len(posts)]
        point.reverse()

        return render(request, 'polls/results.html', {'point': point})
