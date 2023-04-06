//date: 2023-04-06T17:04:03Z
//url: https://api.github.com/gists/e35cd18013d8a41aa4e6cbae5f7cfd93
//owner: https://api.github.com/users/iamsabhoho

func (i *Gemini) Add(id uint64, vector []float32) error {

    ...
   
    if (i.first_add) {
        // First time adding to this index

        // Prepare the float array for saving
        farr := make([][]float32, 1)
        farr[0] = vector

        // Append vector to the numpy file store via memmapping 
        new_count, _, err := Numpy_append_float32_array( i.db_path, farr, int64(dim), int64(1) )
        if err!=nil {
            return errors.Errorf("There was a problem adding to the index backing store file.")
        }
        if new_count!=1 {
            return errors.Errorf("Appending array to local file store did not yield expected result.")
        }

        i.first_add = false
        i.count = 1
        i.dim = dim

    } else {
        ...
	
    return nil

}