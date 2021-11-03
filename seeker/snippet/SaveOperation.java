//date: 2021-11-03T17:08:50Z
//url: https://api.github.com/gists/716f9698a9f601846cb7cfb0a7fd59f5
//owner: https://api.github.com/users/GaetanoPiazzolla

@Autowired
private PostEmbedRepository embeddedRepo;
@Autowired
private PostLinkRepository linkedRepo;
@Autowired
private CommentRepository commentRepo;

public void savePostEmbedded() {

  PostEmbed post = new PostEmbed();
  post.setText("This is an embedded POST");

  post.setComments(new ArrayList<>());
  Comment c = new Comment();
  c.setAuthor("author");
  c.setText("text");
  post.getComments().add(c);

  embeddedRepo.save(post);

}

public void savePostLinked() {

  PostLink post = new PostLink();
  post.setText("This is a linked POST");
  post = linkedRepo.save(post);

  List<Comment> comments = new ArrayList<>();
  Comment c = new Comment();
  c.setAuthor("author");
  c.setText("text");
  c.setPostId(post.getId());
 
  commentRepo.saveAll(comments);

}