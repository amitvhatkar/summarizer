import sys
import os
from summarizer_app.models.summarization_wrapper import summarize_text_using_file
from django.shortcuts import render, HttpResponse



context = {}
ext_list = ['', 'txt']
# Create your views here.
def index(request):
    return render(request, 'index.html')

# def index(request):
#     return HttpResponse('THis is home page')

def about(request):
    return HttpResponse('THis is About')

def news_domain(request):
    return HttpResponse('THis is news domain')

def financial_domain(request):
    return HttpResponse('THis is financial domain')

def general_domain(request):
    return HttpResponse('THis is genral domain')


def generate_summary(text):
    global context
    context['no_file_alert_visibality'] = 'invisible'
    context['input_text'] = text
    summary_result = summarize_text_using_file(text)

    print("----------", summary_result[0])

    if summary_result[0] != -1:
        context['kg_summary'] = summary_result[0]
        context['summary_1_visibality'] = 'visible'
    else:
        context['summary_1_visibality'] = 'invisible'

    if summary_result[2] != -1:
        context['tr_summary'] = summary_result[2]
        context['summary_2_visibality'] = 'visible'
    else:
        context['summary_2_visibality'] = 'invisible'

    if summary_result[1] != -1:
        context['ukg_summary'] = summary_result[1]
        context['summary_3_visibality'] = 'visible'
    else:
        context['summary_3_visibality'] = 'invisible'
    
    if summary_result[0] == -1 and summary_result[1] == -1 and summary_result[2] == -1:
        context['no_file_alert_visibality'] = 'visible'
        context['alert_type'] = 'secondary'
        context['alert_text'] = f'sorry, Summary text can not be summarized!!!'
        context['summary_visibality'] = 'invisible'
        context['summary_1_visibality'] = 'invisible'
        context['summary_2_visibality'] = 'invisible'
        context['summary_3_visibality'] = 'invisible'
        return
    
    context['no_file_alert_visibality'] = 'visible'
    context['alert_type'] = 'primary'
    context['alert_text'] = f'Summarized!!!'
    
def summarize_text(request):
    global context
    print("Summarizer invoded", request.method)
    context = {}  
    
    input_text = request.POST.get('input_text')
    if not input_text:
        print("No text available")
        context['no_file_alert_visibality'] = 'visible'
        context['alert_type'] = 'warning'
        context['alert_text'] = 'Please enter text or select text file!!!'
        context['summary_visibality'] = 'invisible'
        context['summary_1_visibality'] = 'invisible'
        context['summary_2_visibality'] = 'invisible'
        context['summary_3_visibality'] = 'invisible'
    else:
        context['no_file_alert_visibality'] = 'invisible'
        generate_summary(input_text)
    return render(request, 'index.html', context)

def upload_file(request):
    global context
    print(request.method)
    context = {}
    if request.method == "POST":
        # print("---------------------",request.FILES)
        if len(request.FILES) == 0:
            context['no_file_alert_visibality'] = 'visible'
            context['alert_text'] = 'Please select text file or enter text!!!'
            context['alert_type'] = 'warning'
            
            context['summary_visibality'] = 'invisible'
            context['summary_1_visibality'] = 'invisible'
            context['summary_2_visibality'] = 'invisible'
            context['summary_3_visibality'] = 'invisible'
            return render(request, 'index.html', context)
        
        uploaded_file = request.FILES['uploaded_file']
        if not all(ext in uploaded_file.name.lower() for ext in ext_list):
            context['no_file_alert_visibality'] = 'visible'
            context['alert_text'] = 'Please select text file or enter text!!!'
            context['alert_type'] = 'danger'

            context['summary_visibality'] = 'invisible'
            context['summary_1_visibality'] = 'invisible'
            context['summary_2_visibality'] = 'invisible'
            context['summary_3_visibality'] = 'invisible'
            return render(request, 'index.html', context)

        
        input_text = uploaded_file.read().decode("utf-8")
        generate_summary(input_text)        
        
    return render(request, 'index.html', context)