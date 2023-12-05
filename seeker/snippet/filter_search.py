#date: 2023-12-05T16:47:17Z
#url: https://api.github.com/gists/05305a0c6178598be627333c8a5f5cde
#owner: https://api.github.com/users/neodisk17

from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.permissions import AllowAny
from todo.models import TodoModel
from todo.serializers import TodoSerializer


class TodoView(ListCreateAPIView, RetrieveUpdateDestroyAPIView):
    queryset = TodoModel.objects.all()
    serializer_class = TodoSerializer
    permission_classes = [AllowAny]
    filterset_fields = ['is_completed', ]
    search_fields = ['task_name']
    ordering_fields = ['id',]

    def get(self, request, pk=None):
        if pk:
            return self.retrieve(request, pk)
        else:
            return self.list(request)

