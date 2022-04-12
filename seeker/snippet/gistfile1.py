#date: 2022-04-12T16:56:00Z
#url: https://api.github.com/gists/f8a3d2e072126fb321d8a591804ec888
#owner: https://api.github.com/users/Demonoflight

class testAPIView(APIView):
    def get(self, request):
        try:
            if request.data['product_name'] != None:
                lst = TestCls.objects.filter(product_name = request.data['product_name']).values()
               
            
            elif request.data['id'] != None:
                lst = TestCls.objects.filter(id = request.data['id']).values()
                
            
            elif request.data['p_id'] != None:
                lst = TestCls.objects.filter(p_id = request.data['p_id']).values()
                
        except:
            lst = TestCls.objects.all().values()
        return Response({'response': list(lst)})
    
           