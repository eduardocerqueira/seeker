#date: 2024-11-28T17:06:06Z
#url: https://api.github.com/gists/040668a3d82512ba010c62861615c58f
#owner: https://api.github.com/users/anh-tong

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def chunkwise_op(
    k: Float[Array, "chunk_size dim"],
    debug: bool = False,
):
    chunk_size = k.shape[0]
    num_anti_diags = 2 * k.shape[0] - 1

    two_ago = jnp.zeros_like(k)
    one_ago = jnp.zeros_like(k)
    diag = jnp.zeros_like(k)

    # set boundary condition
    two_ago = two_ago.at[0].set(k[0])  # w(0,0), 0, ..., 0
    one_ago = one_ago.at[0].set(k[1])  # w(1,0) and w(0,1), 0, ..., 0
    diag = diag.at[0].set(k[0])  # w(s,s): w(0,0), 0, ..., 0

    inv_k = k[::-1, :]

    def body_fun_phrase_1(i, carry):
        # Phrase 1: operate on area marked with 1
        #  1 0 0 0 0
        #  1 1 0 0 0
        #  1 1 1 0 0
        #  1 1 0 0 0
        #  1 0 0 0 0

        two_ago, one_ago, diag = carry

        corner, right, down = two_ago[:-1], one_ago[:-1], one_ago[1:]

        # w(s, t + 1) + w(s + 1, t) - w(s, t)
        a = right + down - corner

        # to compute <k_s, k_t>, we use the trick of reversing the order of k
        # k_s: k[0], k[1], ..., k[s]
        # k_t: k[i], k[i - 1], ..., k[i - t]
        k_s = k
        k_t = jnp.roll(inv_k, i - 1, axis=0)
        k_s_k_t = jnp.sum(k_s * k_t, axis=-1, keepdims=True)

        # similarly, to compute <k_s, k_{t+1}>
        k_t_plus_1 = jnp.roll(inv_k, i, axis=0)
        k_s_k_t_plus_1 = jnp.sum(k_s * k_t_plus_1, axis=-1, keepdims=True)

        b = diag * (k_s_k_t_plus_1 - k_s_k_t)
        # finally, compute w(s + 1, t + 1)
        next = a + b[:-1]

        # make sure boundary condition is set correctly
        next = jnp.concatenate([jnp.expand_dims(k[i], axis=0), next], axis=0)

        # update diag w(s,s)
        diag = jax.lax.select(i % 2 == 0, diag.at[i // 2].set(next[i // 2]), diag)

        # make sure that lower triangular part of the matrix is zero
        cond = jnp.arange(diag.shape[0]) > i // 2
        cond = jnp.broadcast_to(cond[:, None], next.shape)
        next = jnp.where(cond, 0, next)
        diag = jnp.where(cond, 0, diag)

        two_ago = one_ago
        one_ago = next

        return two_ago, one_ago, diag

    def body_fun_phrase_2(i, carry):
        # Phrase 2: operate on area marked with 1
        #  0 1 1 1 1
        #  0 0 1 1 0
        #  0 0 0 0 0
        #  0 0 0 0 0
        #  0 0 0 0 0
        two_ago, one_ago, diag = carry

        # `corner` now is different
        corner, right, down = two_ago[1:], one_ago[:-1], one_ago[1:]

        # w(s, t + 1) + w(s + 1, t) - w(s, t)
        a = right + down - corner

        # to compute <k_s, k_t>, we use the trick of reversing the order of k
        # ONLY k_s is shifted as i increases
        shift = i - chunk_size
        k_s = jnp.roll(k, -shift, axis=0)
        k_t = jnp.roll(inv_k, -1, axis=0)
        k_s_k_t = jnp.sum(k_s * k_t, axis=-1, keepdims=True)

        # similarly, to compute <k_s, k_{t+1}>
        # k_t does not change
        k_t_plus_1 = inv_k
        k_s_k_t_plus_1 = jnp.sum(k_s * k_t_plus_1, axis=-1, keepdims=True)

        # `diag` is also shifted as i increases when computing w(s + 1, t + 1)
        b = jnp.roll(diag, chunk_size - i, axis=0) * (k_s_k_t_plus_1 - k_s_k_t)
        # finally, compute w(s + 1, t + 1)
        next = a + b[:-1]

        # NO boundary condition is set here but append with zeros
        next = jnp.concatenate([next, jnp.zeros((1, next.shape[-1]))], axis=0)
        # update diag w(s,s)
        next_index = (num_anti_diags - i - 1) // 2
        diag = jax.lax.select(
            i % 2 == 0,
            diag.at[i // 2].set(next[next_index]),
            diag,
        )

        # make sure that lower triangular part of the matrix is zero
        cond_next = jnp.arange(next.shape[0]) > next_index
        cond_next = jnp.broadcast_to(cond_next[:, None], next.shape)
        next = jnp.where(cond_next, 0, next)

        cond_diag = jnp.arange(diag.shape[0]) > i // 2
        cond_diag = jnp.broadcast_to(cond_diag[:, None], diag.shape)
        diag = jnp.where(cond_diag, 0, diag)

        two_ago = one_ago
        one_ago = next

        return two_ago, one_ago, diag

    if debug:
        # write unrolled version of the loop for debugging
        for i in range(2, chunk_size + 1):
            two_ago, one_ago, diag = body_fun_phrase_1(i, (two_ago, one_ago, diag))
        # boundary condition is added at beginning
        # it's not necessary
        one_ago = jnp.roll(one_ago, -1, axis=0)
        for i in range(chunk_size + 1, num_anti_diags):
            two_ago, one_ago, diag = body_fun_phrase_2(i, (two_ago, one_ago, diag))
    else:
        two_ago, one_ago, diag = jax.lax.fori_loop(
            2,
            chunk_size + 1,
            body_fun_phrase_1,
            (two_ago, one_ago, diag),
        )
        one_ago = jnp.roll(one_ago, -1, axis=0)
        two_ago, one_ago, diag = jax.lax.fori_loop(
            chunk_size + 1,
            num_anti_diags,
            body_fun_phrase_2,
            (two_ago, one_ago, diag),
        )

    return diag


def chunkwise_op_ref(k: Float[Array, "chunk_size dim"]):
    """Dummy implementation of chunkwise_op for reference"""

    ret = jnp.zeros((k.shape[0], k.shape[0], k.shape[1]))

    # set boundary condition
    ret = ret.at[0, :].set(k)

    for i in range(1, k.shape[0]):
        for j in range(i, k.shape[0]):
            # w(s, t + 1) + w(s + 1, t) - w(s, t) + w(s,s)*(<k_s, k_t+1> - <k_s, k_t>)
            a = ret[i - 1, j] + ret[i, j - 1] - ret[i - 1, j - 1]
            b = ret[i - 1, i - 1] * (jnp.dot(k[i - 1], k[j]) - jnp.dot(k[i - 1], k[j - 1]))
            ret = ret.at[i, j].set(a + b)

    return ret.diagonal(axis1=0, axis2=1)


if __name__ == "__main__":
    chunk_size = 123
    dim = 15

    k = jax.random.normal(jax.random.key(0), (chunk_size, dim))
    k = k / k.max()

    # test the correctness of the implementation
    output = chunkwise_op(k, debug=False)
    output_unroll = chunkwise_op(k, debug=True)

    assert jnp.allclose(output.squeeze(), output_unroll.squeeze())

    output_ref = chunkwise_op_ref(k)

    assert jnp.allclose(output, output_ref.T)