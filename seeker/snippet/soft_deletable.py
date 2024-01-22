#date: 2024-01-22T16:49:31Z
#url: https://api.github.com/gists/ec98a3fc034d5cc694cd2e38ad391919
#owner: https://api.github.com/users/magyarn

class SoftDeletableqQuerySet(models.QuerySet):
 
    def deleted(self):
        return self.exclude(deleted_at__isnull=True)
 
    def available(self):
        return self.exclude(deleted_at__isnull=False)
 
 
class SoftDeletableManager(models.Manager):
 
    def get_queryset(self):
        return SoftDeletableQuerySet(self.model, using=self._db)
 
    def deleted(self):
        return self.get_queryset().deleted()
 
    def available(self):
        return self.get_queryset().available()
 
 
class SoftDeletableModel(models.Model):
     deleted_at = models.DateTimeField(null=True, default=None)
     
     objects = SoftDeleteManager()
 
     def soft_delete(self):
          self.deleted_at = timezone.now()
          self.save()
 
     def restore(self):
          self.deleted_at = None
          self.save()
 
     def soft_delete_related_records(self):
          print(“Soft deletion for related records is not implemented.”)
 
     def restore_related_records(self):
          print(“Restoration for soft-deleted related records is not implemented.”)
 
 
     class Meta:
          abstract = True
