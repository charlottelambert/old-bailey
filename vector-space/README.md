# thesis/vector-space

The following table shows all files present in the `thesis/vector-space` directory. Below the table there are descriptions of how to run each file.

Name | Contents
-------|-------
`dump_w2v.py` | Code called in `train_embedding_model.py` for writing `tsv` and `txt` data used by projector.tensorflow.org
`README.md` | This file
`train_embedding_model.py` | File for running Word2Vec and FastText models

## Word2Vec Model (need to update)

To run a Word2Vec model, use `train_embedding_model.py`. This will output relevant files to `/work/clambert/models` in a time-stamped directory. You can filter out the top `n` words if you wish to visualize data using projector.tensorflow.org.

Before visualizing data, you must run the following line of code using the path to the saved model in order to output necessary files for projector.tensorflow.org:
```
./dump-w2v.py MODEL_PATH
```
This code will generate two tsv files in the same directory as the input model: `model.tsv` and `labels.tsv`. These two files can then be used to load data for visualization in projector.tensorflow.org.


## FastText MODEL (need to update)

To run a FastText model, use the same process described in the above section for Word2Vec and include the flag `-f`.

