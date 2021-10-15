#date: 2021-10-15T17:06:53Z
#url: https://api.github.com/gists/b1682abb0df431a294d48901cb941993
#owner: https://api.github.com/users/bjudson

"""
Context:
The following code defines two Django models with a many-to-many through relationship, where each
of the three models (including the join model) inherit from a SoftDeletionModel with a custom manager
that uses get_queryset to transparently filter out the soft-deleted objects.

Problem:
If we have a Title instance with a Contributor, then call `title.contributors.remove(contributor)`,
this will set the `deleted_at` field of the TitleContributor join instance. Now, if we call 
`title.contributors.count()` we will get 1, but `title.titlecontributor_set.count()` will return 0.
This is because Title.contributors (ManyRelatedManager) does not inherit from SoftDeletionManager,
while Title.titlecontributor_set (RelatedManager) does inherit from SoftDeletionManager.
"""

# Manager

class SoftDeletionQuerySet(models.QuerySet):
    def delete(self):
        return super().update(deleted_at=timezone.now())

    def hard_delete(self):
        return super().delete()

    def alive(self):
        return self.filter(deleted_at=None)

    def dead(self):
        return self.exclude(deleted_at=None)

class SoftDeletionManager(models.Manager):
    def __init__(self, *args, **kwargs):
        self.alive_only = kwargs.pop("alive_only", True)
        super().__init__(*args, **kwargs)

    def get_queryset(self):
        if self.alive_only:
            return SoftDeletionQuerySet(self.model).filter(deleted_at=None)
        return SoftDeletionQuerySet(self.model)

    def hard_delete(self):
        return self.get_queryset().hard_delete()


class SoftDeletionMixin(models.Model):
    deleted_at = models.DateTimeField(blank=True, null=True)

    objects = SoftDeletionManager()
    all_objects = SoftDeletionManager(alive_only=False)

    class Meta:
        abstract = True

    def delete(self):
        self.deleted_at = timezone.now()
        self.save()

    def hard_delete(self):
        super().delete()

# Models

class Title(SoftDeletionMixin):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    contributors = models.ManyToManyField(to="Contributor", through="TitleContributor")
    
class Contributor(SoftDeletionMixin):
    name = models.CharField(max_length=200)
    
class TitleContributor(SoftDeletionMixin):
    class ContributorRole(models.TextChoices):
        AUTHOR = "author", "Author"
        ILLUSTRATOR = "illustrator", "Illustrator"

    title = models.ForeignKey(Title, on_delete=models.CASCADE)
    contributor = models.ForeignKey(Contributor, on_delete=models.CASCADE)
    role = models.CharField(choices=ContributorRole.choices, max_length=30)  