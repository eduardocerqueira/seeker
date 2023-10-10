#date: 2023-10-10T17:06:59Z
#url: https://api.github.com/gists/13021a0433301810c3cb3f92f341fad2
#owner: https://api.github.com/users/Zymkazhz

from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.shortcuts import get_object_or_404
from rest_framework import serializers

from contest.models import Contest
from mootcourts.conf.auth import DEFAULT_FROM_EMAIL
from service.api.serializers import (EventSerializer, DocumentSerializer, MessageContestSerializer, InvitationSerializer,
                                     MessageSerializer, TelegramSerializer, TopTeamsSerializer,
                                     TopTeamsSerializerUpdate)
from service.models.document_model import Document
from service.models.event_model import EventConf
from service.models.invitation_model import Invitation
from service.models.message_model import Message
from service.models.teaser_model import Teaser
from service.models.top_teams import TopTeams
from tags.api.serializers import TagsSerializer
from tags.models import Tags
from tg_and_conf.models import TelegramAndConf
from users.api.serializers import TeamSerializer, OrganizerHomeSerializer, OrganizerSerializer, \
    ImageOrBase64Field
from users.models import Profile, Team
from users.tasks import send_email_task
from tg_and_conf.tasks import create_conference_task, create_group_task, update_conference_task
from django.utils.safestring import mark_safe
from mootcourts.conf.boilerplate import BASE_DIR
from mootcourts.conf.http import BASE_URL

class ContestSerializer(serializers.ModelSerializer):
    author_id = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())
    author_mootcourt = serializers.SerializerMethodField()
    events = EventSerializer(many=True, required=True)
    document = DocumentSerializer(many=True, required=False)
    teams = serializers.PrimaryKeyRelatedField(queryset=Team.objects.all(), many=True, required=False)
    tags = serializers.PrimaryKeyRelatedField(queryset=Tags.objects.all(), many=True, required=False)
    arbitrators = serializers.PrimaryKeyRelatedField(queryset=Profile.objects.all(), many=True, required=False)
    senior_arbitrators = serializers.PrimaryKeyRelatedField(queryset=Profile.objects.all(), many=True, required=False)
    premut = serializers.PrimaryKeyRelatedField(queryset=Contest.objects.all(), many=True, required=False, allow_null=True)
    message = MessageSerializer(many=True, required=False)
    image = ImageOrBase64Field(required=False)
    telegram_channel = TelegramSerializer(required=False)
    top_teams = TopTeamsSerializerUpdate(required=False)

    def get_author_mootcourt(self, obj):
        author_mootcourt = obj.author
        return OrganizerSerializer(author_mootcourt).data

    def to_representation(self, instance):
        response = super().to_representation(instance)
        response["message"] = sorted(response["message"], key=lambda x: x["publication_date"], reverse=True)
        return response

    def get_message(self, obj):
        messages = Message.objects.filter(contest=obj).order_by('-publication_date')
        return MessageContestSerializer(messages, many=True).data

    def update(self, instance, validated_data):
        new_event_data = validated_data.pop('events', None)
        new_message_data = validated_data.pop('message', None)
        new_team_ids = validated_data.pop('teams', None)
        new_arbitrator_ids = validated_data.pop('arbitrators', None)
        new_senior_arbitrator_ids = validated_data.pop('senior_arbitrators', None)
        new_document_data = validated_data.pop('document', None)
        new_premut_ids = validated_data.pop('premut', None)
        new_tag_ids = validated_data.pop('tags', None)
        top_teams_data = validated_data.pop('top_teams', None)
        user = self.context['request'].user

        if top_teams_data:
            # Если у Contest уже есть связанный TopTeams, обновляем его
            if instance.top_teams:
                top_teams = instance.top_teams
                # Обновляем каждый атрибут в top_teams_data
                for key, value in top_teams_data.items():
                    setattr(top_teams, key, value)
                top_teams.save()
            # Если у Contest еще нет связанного TopTeams, создаем новый
            else:
                top_teams = TopTeams.objects.create(**top_teams_data)
                instance.top_teams = top_teams
                instance.save()

        existing_tag = {tag.id: tag for tag in instance.tags.all()}
        if existing_tag is not None and new_tag_ids is not None:
            new_tag_ids_set = {tag.id for tag in new_tag_ids}
            for tag_id, tag in existing_tag.items():
                if tag_id not in new_tag_ids_set:
                    instance.tags.remove(tag)
            for tag_id in new_tag_ids:
                tag = get_object_or_404(Tags, pk=int(tag_id.id))
                if tag.id not in existing_tag:
                    instance.tags.add(tag)

        existing_premut = {premut.id: premut for premut in instance.premut.all()}
        if existing_premut is not None and new_premut_ids is not None:
            new_premut_ids_set = {premut.id for premut in new_premut_ids}
            for premut_id, premut in existing_premut.items():
                if premut_id not in new_premut_ids_set:
                    instance.premut.remove(premut)
            for premut_id in new_premut_ids:
                premut = get_object_or_404(Contest, pk=int(premut_id.id))
                if premut.id not in existing_premut:
                    instance.premut.add(premut)

        existing_documents = {document.id: document for document in instance.document.all()}
        if new_document_data is not None:
            new_document_ids = [document_one.get('id', None) for document_one in new_document_data]
            # Удалить события, которые отсутствуют в запросе
            for document_id, document in existing_documents.items():
                if document_id not in new_document_ids:
                    document.delete()
            for document_one in new_document_data:
                document_id = document_one.pop('id', None)
                if document_id in existing_documents:
                    document = existing_documents.get(document_id)
                    for attr, value in document_one.items():
                        setattr(document, attr, value)
                    document.save()
                else:
                    document = Document.objects.create(author=self.context['request'].user, **document_one)
                instance.document.add(document)

        existing_senior_arbitrator = {senior_arbitrator.id: senior_arbitrator for senior_arbitrator in instance.senior_arbitrators.all()}
        if existing_senior_arbitrator is not None and new_senior_arbitrator_ids is not None:
            new_senior_arbitrator_ids_set = {senior_arbitrator.id for senior_arbitrator in new_senior_arbitrator_ids}
            for senior_arbitrator_id, senior_arbitrator in existing_senior_arbitrator.items():
                if senior_arbitrator_id not in new_senior_arbitrator_ids_set:
                    instance.senior_arbitrators.remove(senior_arbitrator)
            for senior_arbitrator_id in new_senior_arbitrator_ids:
                senior_arbitrator = get_object_or_404(Profile, pk=int(senior_arbitrator_id.id))
                content_type = ContentType.objects.get_for_model(senior_arbitrator)
                if senior_arbitrator.id not in existing_senior_arbitrator:
                    if Invitation.objects.filter(content_type=content_type,
                                                 author=user.id,
                                                 object_id=int(senior_arbitrator.id),
                                                 invitation_type='invite',
                                                 invitation_status__in=['accepted', 'rejected', 'sent'],
                                                 mootcourt_request_id=instance.id
                                                 ).exists():
                        pass
                    else:
                        invitation_data = {
                            'invitation_status': 'sent',
                            'invitation_type': 'invite',
                            'type_arbitrator': 'senior',
                            'author': user.id,
                            'mootcourt_request_id': instance.id,
                            'content_object': senior_arbitrator,
                            'content_type': content_type.id,
                            'object_id': int(senior_arbitrator_id.id),
                        }
                        serializer = InvitationSerializer(data=invitation_data)
                        serializer.is_valid(raise_exception=True)
                        serializer.save()

        existing_arbitrator = {arbitrator.id: arbitrator for arbitrator in instance.arbitrators.all()}
        if existing_arbitrator is not None and new_arbitrator_ids is not None:
            new_arbitrator_ids_set = {arbitrator.id for arbitrator in new_arbitrator_ids}
            for arbitrator_id, arbitrator in existing_arbitrator.items():
                if arbitrator_id not in new_arbitrator_ids_set:
                    instance.arbitrators.remove(arbitrator)
            for arbitrator_id in new_arbitrator_ids:
                arbitrator = get_object_or_404(Profile, pk=int(arbitrator_id.id))
                content_type = ContentType.objects.get_for_model(arbitrator)
                if arbitrator.id not in existing_arbitrator:
                    if Invitation.objects.filter(content_type=content_type,
                                                 author=user.id,
                                                 object_id=int(arbitrator.id),
                                                 invitation_type='invite',
                                                 invitation_status__in=['accepted', 'rejected', 'sent'],
                                                 mootcourt_request_id=instance.id).exists():
                        pass
                    else:
                        invitation_data = {
                            'invitation_status': 'sent',
                            'invitation_type': 'invite',
                            'type_arbitrator': 'basic',
                            'author': user.id,
                            'mootcourt_request_id': instance.id,
                            'content_object': arbitrator,
                            'content_type': content_type.id,
                            'object_id': int(arbitrator_id.id),
                        }
                        serializer = InvitationSerializer(data=invitation_data)
                        serializer.is_valid(raise_exception=True)
                        serializer.save()

        existing_teams = {team.id: team for team in instance.teams.all()}
        if existing_teams is not None and new_team_ids is not None:
            new_team_ids_set = {team.id for team in new_team_ids}
            for team_id, team in existing_teams.items():
                if team_id not in new_team_ids_set:
                    instance.teams.remove(team)
            for team_id in new_team_ids:
                team = get_object_or_404(Team, pk=int(team_id.id))
                if not team.verification:
                    raise serializers.ValidationError({"error": f"Team {team_id.id} is not verified"})
                else:
                    content_type = ContentType.objects.get_for_model(team)
                    if team.id not in existing_teams:
                        if Invitation.objects.filter(content_type=content_type,
                                                     author=user.id,
                                                     object_id=int(team.captain.id),
                                                     team_invite_id=team.id,
                                                     invitation_type='invite',
                                                     type_invite_team='team',
                                                     invitation_status__in=['accepted', 'rejected', 'sent'],
                                                     mootcourt_request_id=instance.id).exists():
                            pass
                        else:
                            invitation_data = {
                                'invitation_status': 'sent',
                                'invitation_type': 'invite',
                                'author': user.id,
                                'type_invite_team': 'team',
                                'team_invite_id': team.id,
                                'mootcourt_request_id': instance.id,
                                'content_object': team,
                                'content_type': content_type.id,
                                'object_id': int(team.captain.id),
                            }
                            serializer = InvitationSerializer(data=invitation_data)
                            serializer.is_valid(raise_exception=True)
                            serializer.save()

        existing_messages = {message.id: message for message in instance.message.all()}
        if new_message_data is not None:
            new_message_ids = [message_one.get('id', None) for message_one in new_message_data]
            # Удалить события, которые отсутствуют в запросе
            for message_id, message in existing_messages.items():
                if message_id not in new_message_ids:
                    message.delete()
            for message_one in new_message_data:
                message_id = message_one.pop('id', None)
                file_documents_data = message_one.pop('document', None)
                if message_id in existing_messages:
                    message = existing_messages.get(message_id)
                    for attr, value in message_one.items():
                        setattr(message, attr, value)
                    message.save()
                else:
                    message = Message.objects.create(author=self.context['request'].user, **message_one)
                existing_documents = {doc.id: doc for doc in message.document.all()}
                if file_documents_data:
                    new_document_ids = [file_document_data.get('id', None) for file_document_data in
                                        file_documents_data]
                    # Удалить документы, которые отсутствуют в запросе
                    for document_id, document in existing_documents.items():
                        if document_id not in new_document_ids:
                            document.delete()
                    for file_document_data in file_documents_data:
                        file_document_id = file_document_data.pop('id', None)
                        if file_document_id in existing_documents:
                            file_document = existing_documents.get(file_document_id)
                            for attr, value in file_document_data.items():
                                setattr(file_document, attr, value)
                            file_document.save()
                        else:
                            file_document = Document.objects.create(**file_document_data)
                        message.document.add(file_document)
                instance.message.add(message)

        existing_events = {event.id: event for event in instance.events.all()}
        if new_event_data is not None:
            new_event_ids = [event_one.get('id', None) for event_one in new_event_data]
            # Удалить события, которые отсутствуют в запросе
            for event_id, event in existing_events.items():
                if event_id not in new_event_ids:
                    event.delete()
            for event_one in new_event_data:
                event_id = event_one.pop('id', None)
                file_documents_data = event_one.pop('file_document', None)
                if event_id in existing_events:
                    event = existing_events.get(event_id)
                    old_type = event.type
                    conf_id = event.conference_id
                    url_conf = event.url_conference
                    for attr, value in event_one.items():
                        if value is not None or hasattr(event, attr):
                            setattr(event, attr, value)
                    event.conference_id = conf_id
                    event.url_conference = url_conf
                    event.save()
                    new_type = event.type
                    if old_type == 'online' and new_type == 'online':
                        start_event = event.start_event.isoformat()
                        end_event = event.end_event
                        conference_id = event.conference_id
                        if end_event:
                            end_event.isoformat()
                        title_event = event.title
                        description_event = event.description
                        task_kwargs = {
                            'event_id': event.id,
                            'start_event': start_event,
                            'title_event': title_event,
                            'conference_id': conference_id,
                        }

                        if description_event:
                            task_kwargs['description_event'] = description_event
                        update_conference_task.delay(**task_kwargs)

                    if old_type != 'online' and new_type == 'online':
                        start_event = event.start_event.isoformat()
                        end_event = event.end_event
                        if end_event:
                            end_event.isoformat()
                        title_event = event.title
                        description_event = event.description
                        members_email = user.email
                        task_kwargs = {
                            'event_id': event.id,
                            'start_event': start_event,
                            'title_event': title_event,
                            'members_email': members_email,
                        }

                        if description_event:
                            task_kwargs['description_event'] = description_event
                        create_conference_task.delay(**task_kwargs)
                else:
                    if event.type == 'online':
                        start_event = event_one.get('start_event').isoformat()
                        end_event = event_one.get('end_event', None)
                        if end_event:
                            end_event.isoformat()
                        title_event = event_one.get('title')
                        description_event = event_one.get('description', None)
                        members_email = user.email
                        task_kwargs = {
                            'event_id': event.id,
                            'start_event': start_event,
                            'title_event': title_event,
                            'members_email': members_email
                        }

                        if description_event:
                            task_kwargs['description_event'] = description_event

                        create_conference_task.delay(**task_kwargs)
                    event = EventConf.objects.create(author=self.context['request'].user, **event_one)
                existing_documents = {doc.id: doc for doc in event.file_document.all()}
                if file_documents_data:
                    new_document_ids = [file_document_data.get('id', None) for file_document_data in
                                        file_documents_data]
                    # Удалить документы, которые отсутствуют в запросе
                    for document_id, document in existing_documents.items():
                        if document_id not in new_document_ids:
                            document.delete()
                    for file_document_data in file_documents_data:
                        file_document_id = file_document_data.pop('id', None)
                        if file_document_id in existing_documents:
                            file_document = existing_documents.get(file_document_id)
                            for attr, value in file_document_data.items():
                                setattr(file_document, attr, value)
                            file_document.save()
                        else:
                            file_document = Document.objects.create(**file_document_data)
                        event.file_document.add(file_document)
                instance.events.add(event)

        for attr, value in validated_data.items():
            if not isinstance(instance._meta.get_field(attr), models.ManyToManyField):
                setattr(instance, attr, value)
        instance.save()
        return instance

    def create(self, validated_data):
        tags_data = validated_data.pop('tags', None)
        event_data = validated_data.pop('events', None)
        document_data = validated_data.pop('document', None)
        messages_data = validated_data.pop('message', None)
        senior_arbitrators_data = validated_data.pop('senior_arbitrators', None)
        basic_arbitrators_data = validated_data.pop('arbitrators', None)
        teams = validated_data.pop('teams', None)
        user = self.context['request'].user
        validated_data['author'] = user
        title = validated_data.get('title')
        description = validated_data.get('description')
        nicknames_of_administrators = validated_data.get('nicknames_of_administrators')
        image = validated_data.get('image')
        flag_telegram_channel = validated_data.get('flag_telegram_channel')

        contest = Contest.objects.create(**validated_data)

        if user.profile_type == 'organization':
            validated_data['status'] = 'active'
        else:
            subject = f'A new moot court has been created by the user {user.name}'
            message = mark_safe(f'''
            {BASE_URL}/moot-courts/{contest.id}/<br>
            <br>
            Activate the publication in the admin panel: {BASE_URL}/admin/contest/contest/{contest.id}/change/
            ''')
            from_email = DEFAULT_FROM_EMAIL
            to = [admin.email for admin in Profile.objects.filter(is_admin=True)]
            send_email_task.delay(
                subject=subject,
                message=message,
                email_from=from_email,
                recipient_list=to,
            )

        if teams:
            for team_id in teams:
                team = get_object_or_404(Team, pk=int(team_id.id))
                if not team.verification:
                    raise serializers.ValidationError(
                        {"error": f"Team {team_id.id} is not verified"})
                else:
                    content_type = ContentType.objects.get_for_model(team)
                    if Invitation.objects.filter(content_type=content_type,
                                                 author=user.id,
                                                 object_id=int(team.captain.id),
                                                 invitation_status__in=['accepted', 'rejected', 'sent'],
                                                 mootcourt_request_id=contest.id,
                                                 type_invite_team='team',
                                                 ).exists():
                        raise serializers.ValidationError(
                            {"error": f"Invitation for team {team_id.id} with id {contest.id} has already been sent."})
                    else:
                        invitation_data = {
                            'invitation_status': 'sent',
                            'invitation_type': 'invite',
                            'author': user.id,
                            'type_invite_team': 'team',
                            'team_invite_id': team.id,
                            'mootcourt_request_id': contest.id,
                            'content_object': team,
                            'content_type': content_type.id,
                            'object_id': int(team.captain.id),
                        }
                        serializer = InvitationSerializer(data=invitation_data)
                        serializer.is_valid(raise_exception=True)
                        serializer.save()

        if senior_arbitrators_data:
            for senior_arbitrator_id in senior_arbitrators_data:
                arbitrator = get_object_or_404(Profile, pk=int(senior_arbitrator_id.id))
                content_type = ContentType.objects.get_for_model(arbitrator)
                if Invitation.objects.filter(
                        content_type=content_type, object_id=int(senior_arbitrator_id.id),
                        invitation_type='invite', invitation_status__in=['accepted', 'rejected', 'sent'],
                        mootcourt_request_id=contest.id, type_arbitrator='senior').exists():
                    raise serializers.ValidationError(
                        {"error": f"Invitation for senior arbitrator {senior_arbitrator_id.id} with id {contest.id} has already been sent."})
                invitation_data = {
                    'invitation_status': 'sent',
                    'invitation_type': 'invite',
                    'type_arbitrator': 'senior',
                    'author': user.id,
                    'mootcourt_request_id': contest.id,
                    'content_object': arbitrator,
                    'content_type': content_type.id,
                    'object_id': int(senior_arbitrator_id.id),
                }
                serializer = InvitationSerializer(data=invitation_data)
                serializer.is_valid(raise_exception=True)
                serializer.save()

        if basic_arbitrators_data:
            for basic_arbitrator_id in basic_arbitrators_data:
                arbitrator = get_object_or_404(Profile, pk=int(basic_arbitrator_id.id))
                content_type = ContentType.objects.get_for_model(arbitrator)
                if Invitation.objects.filter(
                        content_type=content_type, object_id=int(basic_arbitrator_id.id),
                        invitation_type='invite', invitation_status__in=['accepted', 'rejected', 'sent'],
                        mootcourt_request_id=contest.id, type_arbitrator='basic').exists():
                    raise serializers.ValidationError(
                        {"error": f"Invitation for basic arbitrator {basic_arbitrator_id.id} with id {contest.id} has already been sent."})
                invitation_data = {
                    'invitation_status': 'sent',
                    'invitation_type': 'invite',
                    'type_arbitrator': 'basic',
                    'author': user.id,
                    'mootcourt_request_id': contest.id,
                    'content_object': arbitrator,
                    'content_type': content_type.id,
                    'object_id': int(basic_arbitrator_id.id),
                }
                serializer = InvitationSerializer(data=invitation_data)
                serializer.is_valid(raise_exception=True)
                serializer.save()
        if tags_data:
            for tag_data in tags_data:
                contest.tags.add(tag_data)
        if document_data:
            for document_one in document_data:
                document_one.pop('id', None)
                document = Document.objects.create(author=self.context['request'].user, **document_one)
                contest.document.add(document)
        if event_data:
            for event_one in event_data:
                event_one.pop('id', None)
                file_documents_data = event_one.pop('file_document', None)
                type_data = event_one.get('type', None)

                event = EventConf.objects.create(author=self.context['request'].user, **event_one)

                if type_data:
                    if type_data == 'online':
                        start_event = event_one.get('start_event').isoformat()
                        end_event = event_one.get('end_event', None)
                        if end_event:
                            end_event.isoformat()
                        title_event = event_one.get('title')
                        description_event = event_one.get('description', None)
                        members_email = user.email
                        task_kwargs = {
                            'event_id': event.id,
                            'start_event': start_event,
                            'title_event': title_event,
                            'members_email': members_email
                        }

                        if description_event:
                            task_kwargs['description_event'] = description_event

                        create_conference_task.delay(**task_kwargs)
                if file_documents_data:
                    for file_document_data in file_documents_data:
                        file_document_data.pop('id', None)
                        document = Document.objects.create(**file_document_data)
                        event.file_document.add(document)

                contest.events.add(event)
        if messages_data:
            for message_data in messages_data:
                message_data.pop('id', None)
                file_documents_data = message_data.pop('file_document', None)

                message = Message.objects.create(author=self.context['request'].user, **message_data)

                if file_documents_data:
                    for file_document_data in file_documents_data:
                        file_document_data.pop('id', None)
                        document = Document.objects.create(author=self.context['request'].user, **file_document_data)
                        message.document.add(document)
                contest.message.add(message)

        telegram_data = TelegramAndConf.objects.first()
        if telegram_data:
            if flag_telegram_channel:
                create_group_task.delay(
                    api_id=telegram_data.api_id,
                    api_hash=telegram_data.api_hash,
                    phone_number=telegram_data.phone_number,
                    password= "**********"
                    title=title,
                    description=description,
                    nicknames_of_administrators=nicknames_of_administrators,
                    image=image.name if image else None,
                    contest_id=contest.id
                )
        return contest

    class Meta:
        model = Contest
        fields = '__all__'


class TeaserSerializer(serializers.ModelSerializer):
    image = ImageOrBase64Field(required=False, allow_null=True)

    class Meta:
        model = Teaser
        fields = (
            "publication_date",
            "title",
            "description",
            "bg_color",
            "image",
            "link"
        )


class ContestRetrieveSerializer(serializers.ModelSerializer):
    author = OrganizerHomeSerializer()
    tags = TagsSerializer(many=True)
    teaser = TeaserSerializer()
    teams = TeamSerializer(many=True)
    senior_arbitrators = OrganizerHomeSerializer(many=True)
    arbitrators = OrganizerHomeSerializer(many=True)
    message = serializers.SerializerMethodField()
    events = EventSerializer(many=True)
    document = DocumentSerializer(many=True)
    telegram_channel = TelegramSerializer(required=False)
    top_teams = TopTeamsSerializer(required=False)

    def get_message(self, obj):
        user = self.context.get('request').user

        if user.id == obj.author.id:
            messages = obj.message.all()
        else:
            messages = obj.message.filter(is_published=True)
        # Используем вложенный сериализатор для представления сообщений
        return MessageContestSerializer(messages, many=True, context=self.context).data


    class Meta:
        model = Contest
        fields = (
            "id",
            "title",
            "post_on_home_page",
            "author",
            "premut",
            "type",
            "tags",
            "description",
            "full_text",
            "message",
            "events",
            "document",
            "anyone_join",
            "flag_telegram_channel",
            "telegram_channel",
            "nicknames_of_administrators",
            "teams",
            "senior_arbitrators",
            "arbitrators",
            "status_contest",
            "status",
            "bg_color",
            "image",
            "teaser",
            "top_teams"
        )


class ContestListSerializer(serializers.ModelSerializer):
    author = OrganizerHomeSerializer()
    tags = TagsSerializer(many=True)
    telegram_channel = TelegramSerializer(required=False)
    message = serializers.SerializerMethodField()

    def get_message(self, obj):
        user = self.context.get('request').user

        if user.id == obj.author.id:
            messages = obj.message.all()
        else:
            messages = obj.message.filter(is_published=True)
        # Используем вложенный сериализатор для представления сообщений
        return MessageContestSerializer(messages, many=True, context=self.context).data

    class Meta:
        model = Contest
        fields = (
            "id",
            "title",
            "post_on_home_page",
            "author",
            "premut",
            "type",
            "tags",
            "description",
            "full_text",
            "message",
            "events",
            "document",
            "anyone_join",
            "flag_telegram_channel",
            "telegram_channel",
            "nicknames_of_administrators",
            "teams",
            "senior_arbitrators",
            "arbitrators",
            "status_contest",
            "status",
            "bg_color",
            "image",
            "teaser",
        )


class ContestHomeSerializer(serializers.ModelSerializer):
    tags = TagsSerializer(many=True)
    author = OrganizerHomeSerializer()
    image = ImageOrBase64Field(required=False, allow_null=True)

    class Meta:
        model = Contest
        fields = (
            'id',
            'title',
            'description',
            'status',
            'author',
            'tags',
            'bg_color',
            'image'
        )
        )
