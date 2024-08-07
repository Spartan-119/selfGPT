from django.contrib import admin

from .models import Index, IndexFile


class IndexFileInline(admin.TabularInline):
    model = IndexFile
    extra = 1
    show_change_link = True


@admin.register(Index)
class IndexAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "created", "modified")
    list_filter = ("created", "modified", "user")
    search_fields = ("name", "user__username")
    inlines = [IndexFileInline]


@admin.register(IndexFile)
class IndexFileAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "index", "created", "modified")
    list_filter = ("created", "modified", "index")
    search_fields = ("original_filename", "index__name")
