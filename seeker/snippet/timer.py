#date: 2024-07-12T16:54:46Z
#url: https://api.github.com/gists/a89faf26652ccf9a10933383ef29c5ea
#owner: https://api.github.com/users/richard-to

@contextmanager
def timer(name):
  """Context manager for timing code blocks with custom name."""
  start_time = time.time()
  yield
  end_time = time.time()
  elapsed_time = (end_time - start_time) * 1000
  print(f"{name} took {elapsed_time:.2f} ms")
