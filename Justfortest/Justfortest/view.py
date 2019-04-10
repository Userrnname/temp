from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.shortcuts import redirect
from django.shortcuts import render
import os
from django.views.decorators.csrf import csrf_exempt
def test(request):
    return render_to_response('index.html')
    # return render('index.html')

@csrf_exempt
def upload_file(request): 
    if request.method == "POST":    # 请求方法为POST时，进行处理 
        myFile = request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None 
        constrait = request.POST['constrait']
        constrait = constrait.strip(',').split(',')
        for i in range(len(constrait)):
            constrait[i] = int(constrait[i])
        # print("lalalla",constrait)
        # constrait = [3000,3000,3000]
        # print("hahah",constrait)
        alpha = request.POST['alpha']
        alpha = float(alpha)
        temprature = request.POST['temprature']
        temprature= float(temprature)
        print(constrait,alpha,temprature)
        if not myFile: 
            return HttpResponse("no files for upload!") 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.path.join(current_dir,os.path.pardir)
        new_dir = os.path.join(current_dir,"data")
        print(current_dir)    
        destination = open(os.path.join(new_dir,myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作 
        # print(destination)
        final_dir = os.path.join(new_dir,myFile.name)
        print(final_dir)
        for chunk in myFile.chunks():      # 分块写入文件 
            destination.write(chunk) 
        destination.close() 
        temp_data = pd.read_excel(final_dir)
        result = calculate.main(constrait,temp_data,len(temp_data),alpha,temprature)
        img = result[1]
        # return HttpResponse("upload over!") 
        imb = base64.b64encode(img)#对plot_data进行编码
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        return render(request, "index.html",{"img":imd})

import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from . import calculate

def paint():
    plt.figure(1)
    plt.scatter([0,0],[1,1])
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    return plot_data
    # plt.show()

def gen_mat(request):
    plot_data = paint()
    imb = base64.b64encode(plot_data)#对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64,"+ims
    return render(request, "index.html",{"img":imd})
