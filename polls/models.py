from django.db import models

# Create your models here.

class Question(models.Model):
	question_text = models.CharField(max_length=200)
	pub_date = models.DateTimeField('date published')

	def __str__(self): # __str__ on Python 3
		return self.question_text

class Choice(models.Model):
	question = models.ForeignKey(Question, on_delete=models.CASCADE)
	choice_text = models.CharField(max_length=200)
	votes = models.IntegerField(default=0)

	def __str__(self): #__str__ on Python 3
		return self.choice_text

class MissInform(models.Model):
	url = models.CharField(max_length=200)
	text = models.CharField(max_length=200)
	gender = models.CharField(max_length=5)
	city = models.CharField(max_length=20)
	def __str__(self):
		return self.url
