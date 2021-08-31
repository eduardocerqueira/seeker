#date: 2021-08-31T13:19:21Z
#url: https://api.github.com/gists/bd46a041d776e1872f89de58eef31858
#owner: https://api.github.com/users/michelkana

best_nb_iterations = np.array([ab_test_scores[:,i].argmax() for i in range(len(tree_depths)) ])
best_test_scores = np.array([ab_test_scores[best_nb_iterations[i],i] for i in range(len(tree_depths)) ])
optimal_tree_depth_idx = best_test_scores.argmax()
optimal_nb_iterations = best_nb_iterations[optimal_tree_depth_idx]
optimal_tree_depth = tree_depths[optimal_tree_depth_idx]
optimal_test_score = ab_test_scores[optimal_nb_iterations, optimal_tree_depth_idx]
optimal_train_score = ab_train_scores[optimal_nb_iterations, optimal_tree_depth_idx]
print('The combination of base learner depth {} and {} iterations achieves the best accuracy {}% on test set \
      and {}% on training set.'.format(optimal_tree_depth, optimal_nb_iterations, round(optimal_test_score*100,5), 
      round(optimal_train_score*100,5)))
