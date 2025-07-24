from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class voter(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    age=models.DecimalField(max_digits=5, decimal_places=0)
    address=models.TextField()
    mobile=models.DecimalField(max_digits=12,decimal_places=0)

class election_name(models.Model):
    name=models.TextField() 
    date=models.DateField(null=True)   
class Party(models.Model):
    name=models.TextField()
    def __str__(self):
        return self.name
class candidate(models.Model):
    election=models.ForeignKey(election_name,on_delete=models.CASCADE)
    party=models.ForeignKey(Party,on_delete=models.CASCADE)
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    age=models.DecimalField(max_digits=5, decimal_places=0)
    address=models.TextField()
    mobile=models.DecimalField(max_digits=12,decimal_places=0)
    
class cast_vote(models.Model):
    election=models.ForeignKey(election_name,on_delete=models.CASCADE)
    candidatename=models.ForeignKey(candidate,on_delete=models.CASCADE)
    voter=models.ForeignKey(voter,on_delete=models.CASCADE)
    