#date: 2021-11-15T16:59:31Z
#url: https://api.github.com/gists/763d3a8d5537a8c5580af38d4e27cf4a
#owner: https://api.github.com/users/LasteExile

from django.contrib.contenttypes.models import ContentType
from taggit.models import Tag
from django.db.models import Q, Count
from django.db import models

from infly_todo.models import Task, Contact, TodoOrgModelConfig
from infly_todo.tables import TasksListTable, TasksListTableFilter
from infly_todo.forms import AddEditTaskForm
from infly_todo.operations.utils import TableFormView
from infly_todo.paginator import CustomPaginator

def speedtest(func):
    def wrapper(*args, **kwargs):
        import time
        before = time.time()
        data = func(*args, **kwargs)
        after = time.time()
        print(after - before)
        return data
    return wrapper



class TasksListView(TableFormView):
    template_name = 'infly_todo/list_tasks.html'
    table_class = TasksListTable
    filterset_class = TasksListTableFilter
    form_class = AddEditTaskForm
    model = Task
    paginator_class = CustomPaginator

    @staticmethod
    def create_ids_list(queryset):
        for i in queryset:
            if i.todo_previous_task.count():
                for j in TasksListView.create_ids_list(i.todo_previous_task.only('id', 'previous_task')):
                    yield j
            yield i.id

    def get_table_pagination(self, *args, **kwargs):
        data = super().get_table_pagination(*args, **kwargs)
        if TodoOrgModelConfig.objects.get(organization=self.organization).config.get('TREE_VIEW'):
            queryset = self.get_queryset().filter(previous_task__isnull=True).order_by('-created_date').prefetch_related('todo_previous_task')
            data['list_ids'] = self.create_ids_list(queryset)
            data['table'] = args[0]
        return data

    def get_queryset(self):
        queryset = self.organization.task_set.all().order_by('-created_date')
        active = self.request.GET.get('active')
        status = self.request.GET.get('status')
        if (active is None or active == 'true' or active == 'unknown') and status is None:
            queryset = queryset.filter(status__range=range(0, 2))
        return queryset

    def get_success_url(self):
        return self.request.build_absolute_uri()

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['initial'] = {
            'organization': self.organization,
        }
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['created_by'] = bool(self.request.GET.get('created_by'))
        # context['assigned_to'] = bool(self.request.GET.get('assigned_to'))
        # active_tags = self.request.GET.get('tags')
        # context['active_tags'] = active_tags.split(', ') if active_tags else []
        # context['periods'] = TasksListTableFilter.PeriodChoices
        # active_period = self.request.GET.get('due_period')
        # context['active_period'] = int(active_period) if active_period else None
        # ct = ContentType.objects.get(app_label='infly_todo', model='task')
        # context['statuses'] = Task.Status.choices
        # active_status = self.request.GET.get('status')
        # context['active_status'] = int(active_status) if active_status else None
        # # context['tags'] = [tag.name for tag in Tag.objects.annotate(tag_items=Count('taggit_taggeditem_items')) \
                # # .order_by('-tag_items').filter(taggit_taggeditem_items__content_type=ct)]
        # active = self.request.GET.get('active')
        # context['inactive_tasks'] = not((active is None or active == 'true' or active == 'unknown') and active_status is None)
        # is_parent = self.request.GET.get('is_parent')
        # context['is_parent'] = is_parent == 'true'

        # bp = self.request.user.bp_user
        # context['current_bp'] = bp
        return context

    def form_valid(self, form):
        bp = self.request.user.bp_user
        task = self.organization.task_set.create(
            title=form.cleaned_data['title'],
            note=form.cleaned_data.get('note'),
            created_by=bp,
            due_date=form.cleaned_data.get('due_date'),
            assigned_to=form.cleaned_data.get('assigned_to'),
            previous_task=form.cleaned_data.get('previous_task_input'),
            business_line=form.cleaned_data.get('business_line'),
        )
        task.tags.add(*form.cleaned_data['tags'])
        return super().form_valid(form)
