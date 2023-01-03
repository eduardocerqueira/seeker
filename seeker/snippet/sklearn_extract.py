#date: 2023-01-03T16:39:34Z
#url: https://api.github.com/gists/322cdb5057dd27c4b878ee5bc8302411
#owner: https://api.github.com/users/kayhman

 def predict(
        self,
        X: ArrayLike,
        output_margin: bool = False,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        with config_context(verbosity=self.verbosity):
            class_probs = super().predict(
                X=X,
                output_margin=output_margin,
                ntree_limit=ntree_limit,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
            )
            if output_margin:
                # If output_margin is active, simply return the scores
                return class_probs

            if len(class_probs.shape) > 1 and self.n_classes_ != 2:
                # multi-class, turns softprob into softmax
                column_indexes: np.ndarray = np.argmax(class_probs, axis=1)  # type: ignore
            elif len(class_probs.shape) > 1 and class_probs.shape[1] != 1:
                # multi-label
                column_indexes = np.zeros(class_probs.shape)
                column_indexes[class_probs > 0.5] = 1
            elif self.objective == "multi:softmax":
                return class_probs.astype(np.int32)
            else:
                # turns soft logit into class label
                column_indexes = np.repeat(0, class_probs.shape[0])
                column_indexes[class_probs > 0.5] = 1

            if hasattr(self, "_le"):
                return self._le.inverse_transform(column_indexes)
            return column_indexes