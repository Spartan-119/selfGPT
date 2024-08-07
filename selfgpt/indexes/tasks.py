import logging

from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404

from config import celery_app
from selfgpt.indexes.helpers import transcribe_video
from selfgpt.users.models import User

# Get an instance of a logger
logger = logging.getLogger(__name__)


@celery_app.task()
def task_create_or_update_index_from_files(index_uuid, file_paths):
    try:
        logger.info(f"task_create_or_update_index_from_files - performed - llama_index: {index_uuid}")
    except:  # noqa
        logger.error(f"task_create_or_update_index_from_files - error checking all files - llama_index: {index_uuid}")


@celery_app.task(bind=True)
def task_vectorize_file(self, file_uuid, mime_type):
    from selfgpt.indexes.models import IndexFile, TaskStatus

    try:
        index_file = get_object_or_404(IndexFile, uuid=file_uuid)
        index_file.vectorize_file(mime_type)
        logger.info(
            f"task_vectorize_file with params file_uuid={file_uuid}, mime_type={mime_type} successfully executed"
        )
        TaskStatus.objects.filter(task_id=self.request.id).update(status="SUCCESS")
    except Exception as e:
        TaskStatus.objects.filter(task_id=self.request.id).update(status="FAILURE")
        logger.error(f"task_vectorize_file with params file_uuid={file_uuid}, mime_type={mime_type} failed with {e}")


@celery_app.task(bind=True)
def task_transcribe_video(self, index_uuid, user_uuid, video_url):
    from selfgpt.indexes.models import Index, IndexFile, TaskStatus

    try:
        user = get_object_or_404(User, uuid=user_uuid)
        index = get_object_or_404(Index, uuid=index_uuid, user=user)

    except Index.DoesNotExist:
        return Exception("Index not exist!")

    try:
        logger.info(f"task_transcribe_video started: {video_url}")
        transcription, video_title = transcribe_video(video_url)

        transcription_file = ContentFile(transcription.encode("utf-8"), name=f"{video_title}.txt")

        logger.info(f"task_transcribe_video creating IndexFile {video_url}")
        IndexFile.objects.create(
            index=index, file=transcription_file, original_filename=f"{video_title}.txt", type="video"
        )
        TaskStatus.objects.filter(task_id=self.request.id).update(status="SUCCESS")
    except Exception as e:
        TaskStatus.objects.filter(task_id=self.request.id).update(status="FAILURE")
        logger.error(f"task_transcribe_video - {e}")
        return e
