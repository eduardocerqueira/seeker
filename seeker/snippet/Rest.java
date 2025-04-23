//date: 2025-04-23T17:11:33Z
//url: https://api.github.com/gists/a660488133587b701a59984748a160d6
//owner: https://api.github.com/users/luanfranciscojr

@PutMapping(“/usuarios/{id}”)

public Usuario atualizarUsuario(@PathVariable Long id, @RequestBody Usuario usuario) {

// Lógica para atualizar um usuário

}

@DeleteMapping(“/usuarios/{id}”)

public void deletarUsuario(@PathVariable Long id) {

// Lógica para deletar um usuário

}