#date: 2026-02-12T17:46:36Z
#url: https://api.github.com/gists/0f8f1a7d3f2d4b62584d5c5ab8d10347
#owner: https://api.github.com/users/dryan

import typing

from django.contrib import admin
from django.core.files.images import ImageFile
from django.db import models
from django.utils.translation import gettext_lazy as _


class RedirectableMixin(models.Model):
    previous_url = models.CharField(max_length=255, editable=False, default="")

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        try:
            url = self.get_absolute_url()
        except AttributeError:
            url = None

        if url and not url == self.previous_url:
            from app.models import Redirect

            redirect = Redirect.objects.filter(path=self.previous_url).first()
            if redirect:
                redirect.destination = url
                redirect.is_system = True
                redirect.save()
            else:
                Redirect.objects.create(
                    path=self.previous_url, destination=url, is_system=True
                )

            Redirect.objects.filter(destination=self.previous_url).update(
                destination=url
            )
            Redirect.objects.filter(path=url).delete()

            self.previous_url = url

        return super().save(*args, **kwargs)
    
    
class SlugMixin(models.Model):
    slug = models.SlugField(
        max_length=255,
        blank=True,
        default="",
    )
    slug_source = "name"

    def save(self, *args, **kwargs):
        if hasattr(self, self.slug_source) and not self.slug:
            self.slug = slugify(getattr(self, self.slug_source))
            count = (
                self._meta.model.objects.filter(slug__iexact=self.slug)
                .exclude(pk=self.pk)
                .count()
            )
            if count:
                self.slug = f"{self.slug}-{count + 1}"
        if self.slug:
            self.slug = self.slug.lower()
        return super().save(*args, **kwargs)

    class Meta:
        abstract = True
