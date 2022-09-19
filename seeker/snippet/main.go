//date: 2022-09-19T17:10:50Z
//url: https://api.github.com/gists/ec99f1a18a793b5da6a9cce0ac891537
//owner: https://api.github.com/users/chrisUsick

type redisKvStore interface {
	Keys(context.Context, string) *redis.StringSliceCmd
	Del(context.Context, ...string) *redis.IntCmd
	Set(context.Context, string, interface{}, time.Duration) *redis.StatusCmd
}

func F(ctx context.Context, rdb redisKvStore) (error) {
  keysRes := rdb.Keys(ctx, "app:*")
  if err := keysRes.Err(); err != nil {
    return err
  }
  
  // code omitted for brevity
  ...
}