//date: 2024-04-23T17:00:21Z
//url: https://api.github.com/gists/a2ccb3bb3eddcc69ab11057e80b6f486
//owner: https://api.github.com/users/toteabe

//...
//FilmController
//...
  @GetMapping("")
	public Page<FilmDTO> findAll(@RequestParam(name = "title", required = false) String title,
												 @RequestParam(name = "description", required = false) String description,
												 Pageable pageable) {
    	logger.debug("REST : GET - findAll");
    	Page<FilmDTO> page = service.findAll(title, description, pageable);
		return page; 
    }
//...
//FilmService
//...
	public Page<FilmDTO> findAll(String title, String description, Pageable pageable) {
		logger.debug("findAll()");
		Page<FilmDTO> filmList = filmCustomRepository.queryFilmByTitleAndDescriptionPageable(
				title, description, pageable);
		return filmList;
	}
//...
//FilmCustomRepository
//...
public Page<FilmDTO> queryFilmByTitleAndDescriptionPageable(String title, String description, Pageable pageable)  {

        String queryCountStr = "select count(*) from Film F";
        String queryStr = "select F from Film F";

        if (title != null && description != null) {
            queryStr+= " where F.title like '%'||:title||'%' and F.description like '%'||:description%||'%'";
            queryCountStr+= " where F.title like '%'||:title||'%' and F.description like '%'||:description%||'%'";
        } else if (title != null) {
            queryStr+= " where lower(F.title) like '%'||:title||'%'";
            queryCountStr+= " where lower(F.title) like '%'||:title||'%'";
        } else if (description != null) {
            queryStr+= " where lower(F.description) like '%'||:description%||'%'";
            queryCountStr+= " where lower(F.description) like '%'||:description%||'%'";
        }

        Query queryCount = em.createQuery(queryCountStr);
        if (title != null && description != null) {
            queryCount.setParameter("title", title.toLowerCase());
            queryCount.setParameter("description", description.toLowerCase());
        } else if (title != null) queryCount.setParameter("title", title.toLowerCase());
        else if (description != null) queryCount.setParameter("description", description.toLowerCase());

        Long conteoTotal = (Long)queryCount.getSingleResult();

        Query query = em.createQuery(queryStr, Film.class);

        if ( pageable.getOffset() < conteoTotal ) {
            query.setFirstResult((int) pageable.getOffset())
                    .setMaxResults(pageable.getPageSize());
        } else {
            query.setFirstResult((int) (conteoTotal -pageable.getPageSize() +1))
                    .setMaxResults(pageable.getPageSize());
        }

        if (pageable.getSort().isSorted()) {
            boolean flagNoFirst = false;
            queryStr += " order by";
            for (Sort.Order order : pageable.getSort()) {
                if (flagNoFirst) queryStr += ",";
                queryStr += " " + order.getProperty() + " " +order.getDirection();
                flagNoFirst = true;
            }
        }

        if (title != null && description != null) {
            query.setParameter("title", title.toLowerCase());
            query.setParameter("description", description.toLowerCase());
        } else if (title != null) query.setParameter("title", title.toLowerCase());
        else if (description != null) query.setParameter("description", description.toLowerCase());

        return new PageImpl<>( Util.<FilmDTO, Film>staticEntityListToDtoList(query.getResultList(), FilmDTO.class), pageable, conteoTotal);
    }
    
  //...
  //Util
  //...
    private static final ModelMapper staticMapper = new ModelMapper();

    public static <DTO, ENTITY> List<DTO> staticEntityListToDtoList(Iterable<ENTITY> entities, Class<DTO> dtoClass) {
        List<DTO> dtoList = new LinkedList<>();
        if (entities != null) {
            for (ENTITY e : entities) {
                dtoList.add(staticEntityToDto(e, dtoClass));
            }
        }
        return dtoList;
    }

    public static <DTO, ENTITY> DTO staticEntityToDto(ENTITY entity, Class<DTO> dtoClass) {
        return Util.staticMapper.map(entity, dtoClass);
        
    }