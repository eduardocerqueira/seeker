#date: 2024-05-08T17:06:49Z
#url: https://api.github.com/gists/96d19b0e7bbd6ff4ed7fa5462ed41ef1
#owner: https://api.github.com/users/meyt

from rest_framework import serializers
from rest_framework.utils import model_meta


class NestedModelSerializer(serializers.ModelSerializer):
    def to_representation(self, instance):
        representation = super().to_representation(instance)
        for field_name, field in self.fields.items():
            if not isinstance(field, serializers.PrimaryKeyRelatedField):
                continue

            v = getattr(instance, field_name)
            if not v:
                continue

            model = getattr(self.Meta, "model")
            info = model_meta.get_field_info(model)

            if field_name not in info.relations:
                continue

            relation_info = info.relations[field_name]

            class Nested(serializers.ModelSerializer):
                class Meta:
                    model = relation_info.related_model
                    fields = "__all__"

            representation[field_name] = Nested().to_representation(v)

        return representation
