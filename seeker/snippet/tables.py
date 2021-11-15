#date: 2021-11-15T16:59:31Z
#url: https://api.github.com/gists/763d3a8d5537a8c5580af38d4e27cf4a
#owner: https://api.github.com/users/LasteExile

class TasksListTable(tables.InflyTable):
    title = django_tables2.LinkColumn('infly_todo:task_detail', args=[django_tables2.utils.A('pk')])
    tag = django_tables2.Column(empty_values=(), verbose_name='Group')
    counter = django_tables2.Column(empty_values=(), verbose_name='Task Counter')
    is_parent = django_tables2.Column(empty_values=(), verbose_name='')

    def render_counter(self, value, record):
        return record.todo_previous_task.count()

    def speedtest(func):
        def wrapper(*args, **kwargs):
            import time
            before = time.time()
            data = func(*args, kwargs.get('value'), kwargs.get('record'))
            after = time.time()
            print(after - before)
            return data
        return wrapper
    
    def render_is_parent(self, value, record):
        if not hasattr(self, "is_parent"):
            self.is_parent = TodoOrgModelConfig.objects.get(organization=self.request.user.infly_info.organization).config.get('TREE_VIEW')

        if self.is_parent:
            index = 0
            while True:
                record = record.previous_task
                if not record:
                    break
                index += 1
            if index:
                return format_html(f'<div name="to_move" value="{index}"></div>')
        return ''

    def render_status(self, value, record):
        base_html = \
            '''<label class="label-icon tooltipped" data-position="top" data-tooltip="{}">
               <i class="material-icons" style="color: {}">{}</i></label>'''
        if value == 'New':
            return format_html(base_html.format('New', '#FF4081', 'flag'))
        elif value == 'In Progress':
            return format_html(base_html.format('In Progress', '#3F51B5', 'model_training'))
        elif value == 'On Hold':
            return format_html(base_html.format('On Hold', '#FFC107', 'pause'))
        elif value == 'Done':
            return format_html(base_html.format('Done', '#00AA8D', 'done'))
        elif value == 'Canceled':
            return format_html(base_html.format('Canceled', '', 'close'))
        elif value == 'Spam':
            return format_html(base_html.format('Spam', '', 'delete_forever'))
        elif value == 'Draft':
            return format_html(base_html.format('Draft', '', 'edit'))
        return value

    @speedtest
    def render_tag(self, value, record):
        return ', '.join([tag.name for tag in record.tags.all()])

    def render_created_date(self, value, record):
        return value.strftime('%I:%M %p %m/%d/%Y')

    class Meta:
        model = models.Task
        exclude = ['id', 'note', 'priority', 'task_list', 'completed_date', 'completed', 'org', 'due_date', 'type',
                   'previous_task', 'business_line']
        sequence = ['is_parent', 'status', 'title', 'tag']


class TasksListTableFilter(tables.InflyTableFilterSet):
    tags = CharFilter(field_name='tags', method='filter_tags')
    PeriodChoices = (
        (0, 'Today'),
        (1, 'Yesterday'),
        (2, 'Last 3 days'),
        (3, 'Last 7 days'),
        (4, 'This week'),
        (5, 'This month'),
    )
    due_period = ChoiceFilter(choices=PeriodChoices, field_name='due_period', method='filter_period')
    active = BooleanFilter(method='filter_active')
    is_parent = BooleanFilter(method='filter_is_parent')

    def speedtest(func):
        def wrapper(*args, **kwargs):
            import time
            before = time.time()
            data = func(*args, **kwargs)
            after = time.time()
            print(after - before)
            return data
        return wrapper

    @speedtest
    def filter_is_parent(self, queryset, name, value):
        return queryset.filter(previous_task__isnull=True)

    @speedtest
    def filter_active(self, queryset, name, value):
        if value:
            return queryset.filter(status__range=range(0, 2))
        return queryset

    @speedtest
    def filter_period(self, queryset, name, value):
        value = int(value)
        now = timezone.now().date()
        period = (
            now,  # today
            now - timedelta(1),  # yesterday
            (now - timedelta(3), now),  # last 3 days
            (now - timedelta(7), now),  # last 7 days
            (now - timedelta(days=now.weekday()), now),  # this week (since Monday)
            (now.replace(day=1), now),  # this month
        )
        if value in [0, 1]:
            return queryset.filter(created_date=period[value])
        else:
            return queryset.filter(created_date__range=period[value])

    @speedtest
    def filter_tags(self, queryset, name, value):
        for i in value.split(', '):
            queryset = queryset.filter(tags__name__in=[i])
        return queryset

    class Meta:
        table = TasksListTable
        exclude = ['completed', 'org', 'task_list']
