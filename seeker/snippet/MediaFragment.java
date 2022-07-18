//date: 2022-07-18T17:19:32Z
//url: https://api.github.com/gists/90890c38afd339aa7c4f511c59b65d9f
//owner: https://api.github.com/users/faanghut

private fun initPlayer(String mediaURI) {
	player = new ExoPlayer.Builder(context).build();
	
	ProgressiveMediaSource mediaSource = new ProgressiveMediaSource.Factory(
                    new CacheDataSource.Factory()
                            .setCache(ApplicationClass.getInstance().simpleCache)
                            .setUpstreamDataSourceFactory(new DefaultHttpDataSource.Factory().setUserAgent("ExoplayerDemo"))
                            .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR)
            ).createMediaSource(MediaItem.fromUri(mediaURI));
	
	playerView.setPlayer(player);
	player.setMediaSource(mediaSource);
	player.prepare();
}