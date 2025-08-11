//date: 2025-08-11T17:16:07Z
//url: https://api.github.com/gists/772731b63569130dfe6edb493583e2b6
//owner: https://api.github.com/users/aybabtme

package teehook

var _ mypkg.Hook = (*Tee)(nil)

func NewTeeHook(hooks ...mypkg.Hook) mypkg.Hook {
	return &Tee{hooks: hooks}
}

type Tee struct {
	hooks []mypkg.Hook
}

func (tee *Tee) OnXyzHappened(ctx context.Context, blablabla any) error {
	for i, hooks := range tee.hooks {
		if err := hooks.OnXyzHappened(ctx, blablabla); err != nil {
			return fmt.Errorf("tee mypkg %d: %w", i, err)
		}
	}
	return nil
}

func (tee *Tee) OnAbcHappened(ctx context.Context) error {
	for i, hooks := range tee.hooks {
		if err := hooks.OnAbcHappened(ctx); err != nil {
			return fmt.Errorf("tee mypkg %d: %w", i, err)
		}
	}
	return nil
}