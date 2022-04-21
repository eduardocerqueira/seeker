#date: 2022-04-21T17:19:52Z
#url: https://api.github.com/gists/0dfb5a05a334e1a3793e08f845ed2b33
#owner: https://api.github.com/users/whitespy

class CourseQuerySet(CoreQuerySet):
    def annotate_by_user_flags(self, user):
        return self.annotate(
            is_user_course_editor=Case(
                When(
                    Exists(
                        OrganizationMember.objects.filter(
                            organization=OuterRef("campus__organization"), user=user, role=UserRole.ADMIN
                        )
                    )
                    | Exists(
                        CourseParticipant.objects.filter(
                            course=OuterRef("pk"), user=user, role=CourseParticipantRole.INSTRUCTOR
                        )
                    ),
                    then=True,
                ),
                default=False,
            ),
            is_user_course_student=Exists(
                CourseParticipant.objects.filter(course=OuterRef("pk"), user=user, role=CourseParticipantRole.STUDENT)
            ),
        )

course_queryset = course_queryset.annotate_by_user_flags(user).filter(
            Q(is_user_course_editor=True) | Q(is_user_course_student=True, status=CourseStatus.PUBLISHED)
        )