from datetime import timedelta

from django.db.models import Q
from django.utils import timezone

from selfgpt.indexes.models import TaskStatus


def running_tasks(request):
    if request.user.is_authenticated:
        # Calculate the time 30 minutes ago from now
        time_threshold = timezone.now() - timedelta(minutes=30)

        # Fetch tasks that are either PENDING or STARTED and created within the last 30 minutes
        tasks = TaskStatus.objects.filter(
            Q(user=request.user) & (Q(status="PENDING") | Q(status="STARTED")) & (Q(created__gte=time_threshold))
        ).order_by("-created")

        return {"running_tasks": tasks}
    return {"running_tasks": None}
