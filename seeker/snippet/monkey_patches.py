#date: 2022-12-16T17:05:14Z
#url: https://api.github.com/gists/0e220e7ce4796235e6fbc8efb1e15611
#owner: https://api.github.com/users/cnk

#################################################################################################################
# Patch the filters in the Reports section (and the Page History) so User filters only show Users from the current Site.
# 2020-11-10 cnk: works with 2.11.1
# 2021-12-09 rrollins: Updated to work with 2.15.1.
#################################################################################################################
# This is our patch for LockedPagesReportFilterSet.
def get_site_specific_locked_by_queryset(request):
    """
    Create our own queryset for LockedPages report
    """
    root_page = Site.find_for_request(request).root_page
    user_pks = set(Page.objects.descendant_of(root_page).values_list('locked_by__pk', flat=True))
    return get_user_model().objects.filter(pk__in=user_pks).order_by('last_name')


class SiteSpecificLockedPagesReportFilterSet(wagtail.admin.views.reports.locked_pages.LockedPagesReportFilterSet):
    # 'locked_by' isn't defined in the original LockedPagesReportFilterSet, but it comes into the form because it's a
    # field on Page, and its in the meta class. We add locked_by here so that we can limit the list of users.
    locked_by = django_filters.ModelChoiceFilter(
        field_name='locked_by', queryset=get_site_specific_locked_by_queryset
    )
wagtail.admin.views.reports.locked_pages.LockedPagesView.filterset_class = SiteSpecificLockedPagesReportFilterSet


# This is our patch for LockedPagesView.get_queryset().
def site_specific_queryset(self):
    pages = (
        UserPagePermissionsProxy(self.request.user).editable_pages()
        | Page.objects.filter(locked_by=self.request.user)
    ).filter(locked=True).specific(defer=True)
    # BEGIN PATCH
    # Limit the listing to Pages on the current Site.
    request = get_current_request()
    if request:
        pages = pages.descendant_of(Site.find_for_request(request).root_page)
    # END PATCH
    self.queryset = pages
    return pages
wagtail.admin.views.reports.locked_pages.LockedPagesView.get_queryset = site_specific_queryset


# This is our patch for WorkflowReportFilterSet.
def get_site_specific_requested_by_queryset(request):
    """
    Modified version of wagtail.admin.filters.get_requested_by_queryset
    """
    root_path = Site.find_for_request(request).root_page.path
    pks = set(WorkflowState.objects.filter(page__path__startswith=root_path).values_list('requested_by__pk', flat=True))
    return get_user_model().objects.filter(pk__in=pks).order_by('last_name')


class SiteSpecificWorkflowReportFilterSet(wagtail.admin.views.reports.workflows.WorkflowReportFilterSet):
    requested_by = django_filters.ModelChoiceFilter(
        field_name='requested_by', queryset=get_site_specific_requested_by_queryset
    )
# Note that we're changing just one attribute on the existing WorkflowView class, rather than replacing it entirely.
wagtail.admin.views.reports.workflows.WorkflowView.filterset_class = SiteSpecificWorkflowReportFilterSet


# This is our patch for SiteHistoryReportFilterSet and PageHistoryReportFilterSet.
def site_specific_get_users_for_filter(request):
    """
    Only show users who have modified pages on the current Site.
    """
    request = request or get_current_request()
    # If we weren't sent the request, and couldn't get it from the middleware, we have to give up and return nothing.
    if not request:
        return []

    root_path = Site.find_for_request(request).root_page.path
    user_pks = set(PageLogEntry.objects.filter(page__path__startswith=root_path).values_list('user__pk', flat=True))
    return get_user_model().objects.filter(pk__in=user_pks).order_by('last_name')


class SiteSpecificSiteHistoryReportFilterSet(wagtail.admin.views.reports.audit_logging.SiteHistoryReportFilterSet):
    user = django_filters.ModelChoiceFilter(field_name='user', queryset=site_specific_get_users_for_filter)
# Note that we're changing just one attribute on the existing LogEntriesView class, rather than replacing it entirely.
wagtail.admin.views.reports.audit_logging.LogEntriesView.filterset_class = SiteSpecificSiteHistoryReportFilterSet


class SiteSpecificPageHistoryReportFilterSet(wagtail.admin.views.pages.history.PageHistoryReportFilterSet):
    user = django_filters.ModelChoiceFilter(field_name='user', queryset=site_specific_get_users_for_filter)
# Note that we're changing just one attribute on the existing PageHistoryView class, rather than replacing it entirely.
wagtail.admin.views.pages.history.PageHistoryView.filterset_class = SiteSpecificPageHistoryReportFilterSet


#################################################################################################################
# Patch the Site History log entry classes so that they don't leak page, snippet, or other model
# changes across Sites. Ideally we would like to show ModelLogEntries for all models on a site, but
# that involves a gigantic UNION ALL query across multiple models. So we are settling for only
# showing model history to superusers who could see everything anyway.
# 2021-12-09 rrollins: originally written for Wagtail 2.15.
#################################################################################################################
def site_specific_base_viewable_by_user(self, user):
    if user.is_superuser:
        return self.all()
    else:
        return self.none()
wagtail.core.models.audit_log.BaseLogEntryManager.viewable_by_user = site_specific_base_viewable_by_user


# If we are not showing ModelLogEntries, only show "Page" in our list of possible content types
def site_specific_get_content_types_for_filter():
    request = get_current_request()
    content_type_ids = set()
    for log_model in log_action_registry.get_log_entry_models():
        if log_model.__name__ == 'PageLogEntry' or (request and request.user.is_superuser):
            content_type_ids.update(log_model.objects.all().get_content_type_ids())

    return ContentType.objects.filter(pk__in=content_type_ids).order_by('model')
wagtail.admin.views.reports.audit_logging.get_content_types_for_filter = site_specific_get_content_types_for_filter


def site_specific_page_viewable_by_user(self, user):  # noqa
    # BEGIN PATCH - We filter the initial Q() by Site.
    root_path = None
    request = get_current_request()
    if not request:
        # If we can't determine the current Site, just do what the original method does.
        q = Q(
            page__in=UserPagePermissionsProxy(user).explorable_pages().values_list('pk', flat=True)
        )
    else:
        root = Site.find_for_request(request).root_page
        root_path = root.path
        q = Q(
            page__in=UserPagePermissionsProxy(user).explorable_pages().descendant_of(root).values_list('pk', flat=True)
        )
    # END PATCH

    root_page_permissions = Page.get_first_root_node().permissions_for_user(user)
    if (
        user.is_superuser
        or root_page_permissions.can_add_subpage() or root_page_permissions.can_edit()
    ):
        # Include deleted entries.
        # BEGIN PATCH
        if not request:
            # If we can't determine the current Site, just do what the original method does.
            q = q | Q(page_id__in=Subquery(
                PageLogEntry.objects.filter(deleted=True).values('page_id')
            ))
        else:
            # Limited deleted pages to those on the current Site.
            q = q | Q(page_id__in=Subquery(
                PageLogEntry.objects.filter(deleted=True, page__path__startswith=root_path).values('page_id')
            ))
        # END PATCH

    return PageLogEntry.objects.filter(q)
wagtail.core.models.PageLogEntryManager.viewable_by_user = site_specific_page_viewable_by_user
