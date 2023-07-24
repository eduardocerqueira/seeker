#date: 2023-07-24T17:06:27Z
#url: https://api.github.com/gists/db0e307905aa5aab86f793227d2be597
#owner: https://api.github.com/users/jakelevi1996

from jutility import util

def diagram_g(n_max):
    table = util.Table(
        util.Column("n",        width=10, title="n"),
        util.Column("g_n",      width=10, title="G(n)"),
        util.Column("g_g_n",    width=10, title="G(G(n))"),
    )
    g = [0]
    table.update(n=0, g_n=g[0], g_g_n=g[g[0]])
    for n in range(1, n_max + 1):
        g_n = n - g[g[n - 1]]
        g.append(g_n)
        table.update(n=n, g_n=g_n, g_g_n=g[g_n])

    return table

def diagram_fm(n_max):
    table = util.Table(
        util.Column("n",        width=10, title="n"),
        util.Column("f_n",      width=10, title="F(n)"),
        util.Column("m_n",      width=10, title="M(n)"),
        util.Column("m_f_n",    width=10, title="M(F(n))"),
        util.Column("f_m_n",    width=10, title="F(M(n))"),
    )
    f = [1]
    m = [0]
    table.update(n=0, f_n=f[0], m_n=m[0])
    for n in range(1, n_max + 1):
        m_n = n - f[m[n - 1]]
        m.append(m_n)
        f_n = n - m[f[n - 1]]
        f.append(f_n)
        table.update(n=n, f_n=f_n, m_n=m_n, m_f_n=m[f_n], f_m_n=f[m_n])

    return table

if __name__ == "__main__":
    diagram_g(100)
    diagram_fm(100)
