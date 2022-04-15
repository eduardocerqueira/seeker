#date: 2022-04-15T16:58:03Z
#url: https://api.github.com/gists/9f33db13a5dc13a661829493209ffb67
#owner: https://api.github.com/users/randomcamel

# I didn't write this, but it's too good not to share.
# See also: https://www.mcsweeneys.net/articles/heres-how-time-works-now.
march () {
	perl -e 'use Date::Parse; use POSIX; my @t = localtime; print strftime ("%a Mar ", @t) . int (1 + 0.5 + ((str2time (strftime ("%Y-%m-%d 3:00", @t)) - str2time ("2020-03-01 3:00")) /(60*60*24))) . strftime (" %X %Z 2020\n", @t);'
}