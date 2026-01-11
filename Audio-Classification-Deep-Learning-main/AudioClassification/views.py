from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from AudioClassification.alm import process_audio_for_alm

media_dir = settings.MEDIA_ROOT


def home(request):
    return render(request, "home.html")


def result(request):
    context = {}
    if request.method == "POST":
        file = request.FILES["wavfile"]
        tmp = file.name
        print("The File Name is --> ", tmp)
        fs = FileSystemStorage()
        name = fs.save(file.name, file)
        audio_path = str(media_dir) + str(name)
        print("File Saved and its path is --> ", audio_path)

        # Run ALM pipeline
        alm = process_audio_for_alm(audio_path)
        context["transcription"] = alm["transcription"]
        context["nlu"] = alm["nlu"]
    return render(request, "result.html", context)
