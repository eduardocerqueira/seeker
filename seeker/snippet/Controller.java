//date: 2022-07-25T17:00:00Z
//url: https://api.github.com/gists/c1d36b07a53b89c8e48c6f5f518495ad
//owner: https://api.github.com/users/joaosazup

@RestController
@RequestMapping("/blogs")
public class BlogController {
    private final BlogRepository blogRepository;

    public BlogController(BlogRepository blogRepository) {
        this.blogRepository = blogRepository;
    }

    @PostMapping
    public ResponseEntity<?> cadastrarBlog(@RequestBody @Valid CadastrarBlogRequest request,
            UriComponentsBuilder uriComponentsBuilder) {
        Blog blog = request.toBlog();
        blogRepository.save(blog);
        URI url = uriComponentsBuilder.path("/blogs/{id}").buildAndExpand(blog.getId()).toUri();
        return ResponseEntity.created(url).build();
    }

    @PostMapping("/{blogId}/artigos")
    public ResponseEntity<?> cadastrarArtigo(@PathVariable Long blogId,
            @RequestBody @Valid CadastrarArtigoRequest request,
            UriComponentsBuilder uriComponentsBuilder) {
        Blog blog = blogRepository.findById(blogId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Blog não cadastrado."));
        Artigo artigo = request.toArtigo();
        blog.adiciona(artigo);
        blogRepository.save(blog);
        URI url = uriComponentsBuilder.path("/blogs/{id}").buildAndExpand(blog.getId()).toUri();
        return ResponseEntity.created(url).build();
    }

    @GetMapping("/{blogId}")
    public ResponseEntity<?> getBlog(@PathVariable Long blogId) {
        Blog blog = blogRepository.findById(blogId).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Blog não cadastrado."));
        return ResponseEntity.ok(blog);
        
    }
}
