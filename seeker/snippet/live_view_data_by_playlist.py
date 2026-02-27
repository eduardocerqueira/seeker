#date: 2026-02-27T17:11:21Z
#url: https://api.github.com/gists/5b68bc215e8523034133857b0049df76
#owner: https://api.github.com/users/ajin-UN

from collections import defaultdict

from google.appengine.ext import ndb

from bf_utils.environment_utils import find_gcs_domain_url, EMAIL_SENDER
from bf_utils.model_utils import chunks
from bf_utils.utils import AbstractBlobHandler
from blinkfire import mail
from blinkfire.user import BfUser, Playlist, FacebookProfileManager
from generic.models.live_stats import FacebookLiveData
from generic_libs.gcs_adaptor import GCSAdaptor
from modules.facebook_poller.graph_api.client import FacebookGraphAPIClient


class JsonBlobHandler(AbstractBlobHandler):
    mime_type = 'json'

    @classmethod
    def _upload(cls, data):
        """ Upload data file from buffer and return blob key """
        bucket = cls.get_bucket_name_based_on_active_environment()
        print("Uploading file to {} bucket...".format(bucket))
        file_url = GCSAdaptor.upload_blob_to_gcs(
            data,
            bucket=bucket,
            content_type=cls.mime_type,
        )
        return file_url

    @classmethod
    def send_json_report(cls, user, email_address, data, subject):
        file_url = cls._upload(data)
        file_url = '{}{}'.format(find_gcs_domain_url(), file_url)
        cls.send_report(user, email_address, subject, file_url)

    @classmethod
    def send_report(cls, user, email_address, subject, file_url):
        print("user {} sending to {} {}".format(user.key.id(), email_address, subject))
        mail.send_mail(sender=EMAIL_SENDER,
                       to=email_address,
                       subject=subject,
                       body=file_url,
                       html='')


class ProcessPlaylistLiveViewersMixIn(object):

    acceptable_medium = ['facebook']
    acceptable_variety = ['live']
    page_size = 50

    def __init__(self, playlist_id):
        self.playlist = self.get_playlist(playlist_id)
        self.item_keys = self.playlist.items if self.playlist else []
        self.user = BfUser.get_by_id(5838499713646592) # any superuser

    def get_playlist(self, playlist_id):
        playlist = Playlist.get_by_id(int(playlist_id)) if playlist_id else None
        return playlist

    def process_playlist(self):
        response_dict = {}
        for item_keys_chunk in chunks(self.item_keys, self.page_size):
            items = ndb.get_multi(item_keys_chunk)
            valid_items = [item for item in items
                           if item
                           and item.medium in self.acceptable_medium
                           and item.variety in self.acceptable_variety]
            if not valid_items:
                continue

            print('Processing {} items...'.format(len(valid_items)))
            items_by_channel_key = defaultdict(list)
            for item in valid_items:
                items_by_channel_key[item.channel].append(item)

            channels_by_key = {channel.key: channel for channel in filter(None, ndb.get_multi(items_by_channel_key))}
            for channel_key, items_per_channel in items_by_channel_key.items():
                response_dict = self.process_items_by_channel(channels_by_key.get(channel_key),
                                                              items_per_channel,
                                                              response_dict)
        return response_dict

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"c "**********"h "**********"a "**********"n "**********"n "**********"e "**********"l "**********") "**********": "**********"
        return getattr(self, 'get_{}_access_token'.format(channel.medium), None)(channel)

    @staticmethod
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"f "**********"a "**********"c "**********"e "**********"b "**********"o "**********"o "**********"k "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"c "**********"h "**********"a "**********"n "**********"n "**********"e "**********"l "**********") "**********": "**********"
        access_token, _ = "**********"
        return access_token

    def get_live_data(self, item):
        return getattr(self, 'get_{}_live_data'.format(item.medium), None)(item)

    @staticmethod
    def get_facebook_live_data(item):
        return FacebookLiveData.query(FacebookLiveData.item == item.key).get()

    def process_items_by_channel(self, channel, items, response_dict):
        access_token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            return response_dict

        client = "**********"
        insights_fields = ['post_video_views_live']
        item_ids = [item.key.id() for item in items]
        batch_requests = [
            client.post_insights_config(
                post_id=item_id,
                metrics=insights_fields,
                period='lifetime',
            )
            for item_id in item_ids
        ]
        batch_responses = "**********"
        for item, response in zip(items, batch_responses):
            if not response:
                continue
            try:
                content = response.get_response()
            except Exception as e:
                print('Error processing {}: {}'.format(item.key.id(), repr(e)))
                content = {}

            insights_data = content.get('data')
            if not insights_data:
                post_video_views_live = 'No result from Graph API'
            else:
                post_video_views_live = insights_data[0].get('values', [])[0].get('value', 0)

            live_data = self.get_live_data(item)
            item_dict = {
                'viewers': live_data.viewers if live_data else 'No Live Stats Found in Datastore',
                'sum_of_views': sum(live_data.viewers) if live_data and live_data.viewers else 0,
                'views': item.video_views,
                'post_video_views_live': post_video_views_live,
            }
            print('Including {} - {} to the response dict...'.format(item.key.id(), item_dict))
            response_dict[item.key.id()] = item_dict
        return response_dict

    def send_email(self, email_address):
        response_dict = self.process_playlist()
        print('Finished processing playlist {}...'.format(self.playlist.display_name))
        print('Response dict: {}'.format(response_dict))
        print('Sending email to {}...'.format(email_address))
        subject = 'Live Viewers & Views Data for {}'.format(self.playlist.display_name)
        JsonBlobHandler.send_json_report(self.user, email_address, response_dict, subject)


playlist_id = '6029345935196160' # any playlist id
email_addresses = ['aiai@blinkfire.com', 'scottm@blinkfire.com'] # any admin emails

for email_address in email_addresses:
    if not email_address:
        continue
    ProcessPlaylistLiveViewersMixIn(playlist_id).send_email(email_address)
