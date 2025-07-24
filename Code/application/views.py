from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login,logout,authenticate
from .models import voter,Party,candidate,election_name,cast_vote
# Create your views here.
import datetime
def home(request):
    return render(request,'home.html')
def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  
        else:
            messages.error(request, 'please check the password...')
            return redirect('login')
    return render(request, 'login.html')

def Admin_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if username =='admin' and password=='admin':
            user = authenticate(request,username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home') 
            else:
                user=User.objects.create_user(username=username,password=password,is_active=True,is_staff=True,is_superuser=True) 
                user.save()
                login(request,user)
                return redirect('home')
        else:
            messages.error(request, 'please check the password...')
            return redirect('login')
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('home')

def voter_registration(request):
    if request.method == "POST":
        name = request.POST.get("name")
        age = request.POST.get("age")
        address = request.POST.get("address")
        username = request.POST.get("username")
        password = request.POST.get("password")
        mobile=request.POST.get('mobile')
        user=User.objects.filter(username=username)
        if user:
            messages.success(request,'username already taken')
        else:
            user = User.objects.create_user(username=username, password=password,first_name=name)
            user.save()
            u=User.objects.get(username=username)
            v=voter.objects.create(user=u,mobile=mobile,address=address,age=age)
            v.save()
            return redirect('voter_registration')
    return render(request,'voterregister.html')

def candidate_registration(request):
    party=Party.objects.all()
    election=election_name.objects.all()
    if request.method == "POST":
        name = request.POST.get("name")
        age = request.POST.get("age")
        address = request.POST.get("address")
        username = request.POST.get("username")
        password = request.POST.get("password")
        party_id=request.POST.get('party_id')
        election_id=request.POST.get('election_id')
        party=Party.objects.get(id=party_id)
        election=election_name.objects.get(id=election_id)
        mobile=request.POST.get('mobile')
        user=User.objects.filter(username=username)
        if user:
            messages.success(request,'username already taken')
        else:
            u=User.objects.create_user(username=username,password=password,first_name=name,is_staff=True)
            u.save()
            user=User.objects.get(username=username)
            s=candidate.objects.create(user=u,party=party,election=election,age=age,address=address,mobile=mobile)
            s.save()
            messages.success(request,'candidate saved sucessfully')
        return redirect('voter_registration')
    return render(request, "candidateregistration.html",{'party':party,'election':election})

def voter_list(request):
    voters = voter.objects.all()
    return render(request, "voter_list.html", {"voters": voters})

def add_party(request):
    if request.method == "POST":
        name = request.POST.get("name")
        if name:
            Party.objects.create(name=name)
            return redirect("add_party")  
    return render(request, "addparty.html")

def add_election(request):
    if request.method == "POST":
        name = request.POST.get("name")
        date=request.POST.get('date')
        if name:
            election_name.objects.create(name=name,date=date)
            return redirect("add_election") 
    return render(request, "addelection.html")

def elections(request):
    election=election_name.objects.all().order_by('-id')
    today=datetime.date.today()
    return render(request,'select_election.html',{'elections':election,'today':today})

def cast_vote_made(request, pk):
    election = election_name.objects.get(id=pk)
    candidates = candidate.objects.filter(election=election)
    return render(request, "candidatelist.html", {"candidates": candidates, "election": election})

def vote_made(request):
    if request.method == "POST":
        candidatename_id = request.POST.get("candidatename_id")
        election_id = request.POST.get("election_id")
        election=election_name.objects.get(id=election_id )
        voter_instance = voter.objects.get(user=request.user)
        selected_candidate = candidate.objects.get(id=candidatename_id)
        if not cast_vote.objects.filter(election=election, voter=voter_instance).exists():
            cast_vote.objects.create(election=election, candidatename=selected_candidate, voter=voter_instance)
            messages.success(request,'successfully Vote Done')
            return redirect("elections") 
        else:
            messages.success(request,'You already Caste Your Vote')
            return redirect("elections")
from django.db.models import Count
def result(request,pk):
    election = election_name.objects.get(id=pk)
    results = cast_vote.objects.filter(election=election).values("candidatename").annotate(vote_count=Count("candidatename"))
    results = [{"candidatename": candidate.objects.get(id=r["candidatename"]), "vote_count": r["vote_count"]} for r in results]
    return render(request, "voteCount.html", {"results": results})

def view_result(request):
    election=election_name.objects.all().order_by('-id')
    return render(request,'view_election.html',{'elections':election})

def view_candidates(request):
    candidates=candidate.objects.all()
    return render(request,'view_candidates.html',{'candidates':candidates})