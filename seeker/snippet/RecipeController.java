//date: 2023-04-14T17:04:29Z
//url: https://api.github.com/gists/7fb7a54a193dd8ba5791986bbe687eee
//owner: https://api.github.com/users/arijit83sarkar

package com.raven.springbootthymeleafrecipeorganizer.controller;
// imports are removed

@Controller
@RequestMapping("/recipe")
public class RecipeController {
	private final RecipeService recipeService;

	@Autowired
	public RecipeController(RecipeService recipeService) {
		this.recipeService = recipeService;
	}

	@GetMapping("/new")
	public ModelAndView newRecipe() {
		ModelAndView modelAndView = new ModelAndView("recipe/newrecipe");
		modelAndView.addObject("recipe", new Recipe());

		return modelAndView;
	}
  
  	@GetMapping("/list")
	public ModelAndView getRecipeList() {
    		// existing code
	}
}
