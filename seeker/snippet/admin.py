#date: 2025-03-21T17:08:00Z
#url: https://api.github.com/gists/40ce302c33c3286b1f461bfd69458a1a
#owner: https://api.github.com/users/jterjjj65

from django.contrib import admin
from django import forms
from .models import Category, ProductType, Product, AttributeValue, ProductImage, Attribute, AttributeOption

# Inline для AttributeOption (чтобы редактировать варианты прямо на странице атрибута)
class AttributeOptionInline(admin.TabularInline):
    model = AttributeOption
    extra = 1
    fields = ('value', 'display_value', 'order')
    ordering = ('order',)

# Inline для AttributeValue (чтобы редактировать значения атрибутов на странице товара)
class AttributeValueInline(admin.TabularInline):
    model = AttributeValue
    extra = 1
    fields = ('attribute', 'option')

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        # Ограничиваем выбор атрибутов только теми, которые соответствуют типу товара
        if db_field.name == 'attribute':
            product_id = request.resolver_match.kwargs.get('object_id')
            if product_id:
                product = Product.objects.get(id=product_id)
                kwargs['queryset'] = Attribute.objects.filter(product_type=product.product_type)

        # Для поля 'option' пока не добавляем фильтрацию, так как будем использовать JavaScript
        if db_field.name == 'option':
            # Если атрибут ещё не выбран, показываем пустой список
            kwargs['queryset'] = AttributeOption.objects.none()

        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    class Media:
        js = ('admin/js/attribute_value_filter.js',)

# Inline для ProductImage (чтобы редактировать изображения на странице товара)
class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 1
    fields = ('image',)

# Админка для Category
@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'parent', 'is_active')
    list_filter = ('is_active', 'parent')
    search_fields = ('name',)
    ordering = ('name',)

# Админка для ProductType
@admin.register(ProductType)
class ProductTypeAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)
    ordering = ('name',)

# Админка для Attribute
@admin.register(Attribute)
class AttributeAdmin(admin.ModelAdmin):
    inlines = [AttributeOptionInline]
    list_display = ('name', 'product_type')
    list_filter = ('product_type',)
    search_fields = ('name',)
    ordering = ('name',)

# Админка для AttributeOption
@admin.register(AttributeOption)
class AttributeOptionAdmin(admin.ModelAdmin):
    list_display = ('value', 'attribute', 'display_value', 'order')
    list_filter = ('attribute',)
    search_fields = ('value', 'display_value')
    ordering = ('attribute', 'order')

# Админка для Product
@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    inlines = [AttributeValueInline, ProductImageInline]
    list_display = ('name', 'product_type', 'category', 'price', 'is_active', 'created_at')
    list_filter = ('product_type', 'category', 'is_active')
    search_fields = ('name', 'description')
    ordering = ('-created_at',)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('product_type', 'category').prefetch_related('attribute_values', 'images')

# Админка для AttributeValue
@admin.register(AttributeValue)
class AttributeValueAdmin(admin.ModelAdmin):
    list_display = ('product', 'attribute', 'option')
    list_filter = ('attribute',)
    search_fields = ('product__name',)
    ordering = ('product',)

# Админка для ProductImage
@admin.register(ProductImage)
class ProductImageAdmin(admin.ModelAdmin):
    list_display = ('product', 'image')
    search_fields = ('product__name',)
    ordering = ('product',)