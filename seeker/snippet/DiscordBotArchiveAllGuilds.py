#date: 2023-06-02T16:43:41Z
#url: https://api.github.com/gists/845c0b0f7f0ec7977311d0d1654dd82f
#owner: https://api.github.com/users/0187773933

#!/usr/bin/env python3
import requests
from pprint import pprint
from box import Box
from pathlib import Path
import shutil
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
from slugify import slugify # pip install python-slugify

OVERWRITE = False
# OVERWRITE = True

def download_file( options ):
	try:
		if OVERWRITE == False:
			if options[ 1 ].is_file() == True:
				if options[ 1 ].stat().st_size > 1:
					return True
		r = requests.get( options[ 0 ] , stream=True )
		total_size = int( r.headers.get( "content-length" , 0 ) )
		block_size = 1024
		t = tqdm( total=total_size , unit="iB" , unit_scale=True )
		with open( str( options[ 1 ] ) , "wb" ) as f:
			for data in r.iter_content( block_size ):
				t.update( len( data ) )
				f.write( data )
		t.close()
		if total_size != 0 and t.n != total_size:
			print( "ERROR , something went wrong" )
	except Exception as e:
		print( e )

def write_json( file_path , python_object ):
	with open( file_path , 'w', encoding='utf-8' ) as f:
		json.dump( python_object , f , ensure_ascii=False , indent=4 )

def read_json( file_path ):
	with open( file_path ) as f:
		return json.load( f )

def batch_process( options ):
	batch_size = len( options[ "batch_list" ] )
	with ThreadPoolExecutor() as executor:
		result_pool = list( tqdm( executor.map( options[ "function_reference" ] , iter( options[ "batch_list" ] ) ) , total=batch_size ) )
		return result_pool

class DisordChannelArchiverBot:
	def __init__( self , config={} ):
		self.config = Box( config )
		self.headers = 	headers = {
			"accept": "application/json, text/plain, */*" ,
			"Authorization": "**********"
		}

	def enumerate_request( self , limit=100 , after=False ):
		# https://discord.com/developers/docs/resources/channel#get-channel-messages
		# https://discord.com/developers/docs/reference#snowflakes
		pass

	def get_guilds( self ):
		# https://discord.com/developers/docs/resources/user#get-current-user-guilds
		limit = 200
		params = { "limit": limit }
		url = f"https://discord.com/api/users/@me/guilds"
		response = requests.get( url , headers=self.headers , params=params )
		response.raise_for_status()
		self.guilds = response.json()

	def get_guild_channels( self , guild_id ):
		# https://discord.com/developers/docs/resources/guild#get-guild-channels
		params = {}
		url = f"https://discord.com/api/guilds/{guild_id}/channels"
		response = requests.get( url , headers=self.headers , params=params )
		response.raise_for_status()
		result = response.json()
		return result

	def get_channel_messages( self , channel_id ):
		# https://discord.com/developers/docs/resources/channel#get-channel-messages
		limit = 100
		params = { "limit": limit }
		url = f"https://discord.com/api/channels/{channel_id}/messages"
		response = requests.get( url , headers=self.headers , params=params )
		response.raise_for_status()
		messages = response.json()
		# arrives in reverse order , aka latest message = array[0] , first message = array[-1]
		messages.reverse()
		# pprint( messages )
		if len( messages ) < limit:
			return messages
		finished = False
		iterations = 1
		while finished == False:
			print( f"Gathering {limit} new messages , Round = {iterations} , Total = {len( messages )}" )
			params[ "before" ] = messages[ 0 ][ "id" ]
			response = requests.get( url , headers=self.headers , params=params )
			response.raise_for_status()
			new_messages = response.json()
			new_messages.reverse()
			messages = new_messages + messages
			iterations += 1
			if len( new_messages ) < limit:
				finished = True
		# print( len( messages ) )
		return messages

	def download_all_message_attachments( self , output_directory , messages ):
		download_list = []
		total_messages = len( messages )
		# zfill_number = len( str( total_messages ) )
		zfill_number = 3
		item_total = 1
		for message_index , message in enumerate( messages ):
			if "attachments" not in message:
				continue
			if len( message[ "attachments" ] ) < 1:
				# continue
				if "embeds" not in message:
					continue
				if len( message[ "embeds" ] ) < 1:
					continue
				for embed_index , embed in enumerate( message[ "embeds" ] ):
					if "thumbnail" not in embed:
						continue
					if "proxy_url" not in embed[ "thumbnail" ]:
						continue
					file_type = embed[ "thumbnail" ][ "url" ].split( "." )[ -1 ][ 0 : 3 ]
					if file_type == "jpe":
						file_type = "jpeg"
					download_list.append([
						embed[ "thumbnail" ][ "proxy_url" ] ,
						output_directory.joinpath( f'{str(item_total).zfill(zfill_number)}.{file_type}' )
					])
					item_total += 1
			else:
				for attachment_index , attachment in enumerate( message[ "attachments" ] ):
					if "url" not in attachment:
						pprint( message )
						continue
					if "filename" not in attachment:
						pprint( message )
						continue
					download_list.append([
						attachment[ "url" ] ,
						output_directory.joinpath( f'{str(item_total).zfill(zfill_number)}{Path( attachment[ "filename" ] ).suffix}' )
					])
					item_total += 1
		# pprint( download_list )
		batch_process({
			"max_workers": 10 ,
			"batch_list": download_list ,
			"function_reference": download_file
		})

	def get_channel( self , channel_id ):
		params = {}
		url = f"https://discord.com/api/channels/{channel_id}"
		response = requests.get( url , headers=self.headers , params=params )
		response.raise_for_status()
		result = response.json()
		# pprint( result )
		return result

	def archive_channel( self , channel_id , output_base_directory=False , save_json=True ):
		channel = self.get_channel( channel_id )
		if "name" not in channel:
			return False
		if output_base_directory == False:
			output_base_directory = Path.cwd().joinpath( "downloads" , channel[ "name" ] )
		# output_base_directory.mkdir( parents=True , exist_ok=True )
		message_archive_save_path = output_base_directory.joinpath( f'{channel[ "name" ]}.json' )
		attachment_base_directory = output_base_directory.joinpath( channel[ "name" ] )
		# shutil.rmtree( str( attachment_base_directory ) , ignore_errors=True )
		attachment_base_directory.mkdir( parents=True , exist_ok=True )

		print( f"1.) Downloading Message Archive of {channel[ 'name' ]}" )
		messages = self.get_channel_messages( channel_id )
		if save_json == True:
			write_json( str( message_archive_save_path ) , messages )
		print( f"2.) Downloading Attachments from {channel[ 'name' ]}" )
		self.download_all_message_attachments( attachment_base_directory , messages )

	def archive_all( self ):
		self.get_guilds()
		total_guilds = len( self.guilds )
		for guild_index , guild in enumerate( self.guilds ):

			# 1.) Prep Download Folder For Each Guild
			guild_name_slug = slugify( guild[ "name" ] )
			guild_output_dir = self.config.output_dir.joinpath( guild_name_slug )
			guild_output_dir.mkdir( parents=True , exist_ok=True )

			# 2.) Get all the channels in the guild
			g_channels = self.get_guild_channels( guild[ "id" ] )

			# 3.) Find and Sort Channels By Category
			g_categories = { channel[ "id" ]: { "name": channel[ "name" ] , "channels": [] } for channel in g_channels if channel[ "type" ] == 4 }
			for channel in g_channels:
				if channel[ "type" ] == 4:
					continue
				if "parent_id" in channel:
					if channel[ "parent_id" ] in g_categories:
						g_categories[ channel[ "parent_id" ] ][ "channels" ].append( channel )
					else:
						g_categories[ channel[ "id" ] ] = { "name": channel[ "name" ] , "channels": [ channel ] }
			# pprint( g_categories )

			# 4.) Save JSON Structure of Guild
			write_json( guild_output_dir.joinpath( f"{guild_name_slug}.json" ) , g_categories )

			# 5.) Download Each Channel
			total_categories = len( g_categories )
			for category_index , category in enumerate( g_categories ):
				total_channels = len( g_categories[ category ][ "channels" ] )
				for channel_index , channel in enumerate( g_categories[ category ][ "channels" ] ):
					channel_output_dir = guild_output_dir.joinpath( channel[ "name" ] )
					print( f"Downloading === Guild [ {guild_index+1} ] of {total_guilds} || Category [ {category_index+1} ] of {total_categories} || Channel [ {channel_index+1} ] of {total_channels}" )
					self.archive_channel( channel[ "id" ] , output_base_directory=channel_output_dir , save_json=True )


if __name__ == "__main__":
	bot = DisordChannelArchiverBot({
		"token": "**********"
		"output_dir": Path.cwd().joinpath( "DOWNLOAD_ALL" ) ,
	})
	bot.archive_all()hive_all()