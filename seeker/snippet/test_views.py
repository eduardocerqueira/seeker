#date: 2021-10-21T17:00:16Z
#url: https://api.github.com/gists/e9cb18ccd8e0bd94c9f43fdb762926ca
#owner: https://api.github.com/users/ksusonic

    def test_cache(self):
        cache.clear()
        temp_post = Post.objects.create(
            text='123',
            author=self.user
        )
        response = self.authorized.get(INDEX_URL)
        temp_post.delete()

        self.assertEqual(
            response.content,
            self.authorized.get(INDEX_URL).content
        )