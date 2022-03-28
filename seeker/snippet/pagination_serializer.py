#date: 2022-03-28T17:04:52Z
#url: https://api.github.com/gists/69895132024b4a8da82e779df5285b97
#owner: https://api.github.com/users/david-luk4s

from rest_framework.utils.urls import replace_query_param, remove_query_param

class StandarUtilsPagination:

    @staticmethod
    def get_next_link(paginator, page_number, url):
        if not page_number < paginator.num_pages:
            return None
        page_number = paginator.validate_number(page_number + 1)
        return replace_query_param(url, 'page', page_number)

    @staticmethod
    def get_previous_link(paginator, page_number, url):
        if not page_number > 1:
            return None
        page_number = paginator.validate_number(page_number - 1)
        if page_number == 1:
            return remove_query_param(url, 'page')
        return replace_query_param(url, 'page', page_number)

from django.core.paginator import Paginator, EmptyPage

page_number = int(self.request.query_params.get('page',1))
page_size = int(self.request.query_params.get('page_size', 200))
search = self.request.query_params.get('search', '')
qs = get_user_model().objects.all()

try:
    paginator = Paginator(qs, page_size)
    serializer = ModelSerializer(paginator.page(page_number) , many=True, context={'request':request})
except EmptyPage:
    return Response({'detail': 'is no page for searched item'}, status=HTTP_404_NOT_FOUND)

return Response({
    'count': paginator.count,
    'next': StandarUtilsPagination.get_next_link(paginator, page_number, request.build_absolute_uri()),
    'previous': StandarUtilsPagination.get_previous_link(paginator, page_number, request.build_absolute_uri()),
    'results': serializer.data})