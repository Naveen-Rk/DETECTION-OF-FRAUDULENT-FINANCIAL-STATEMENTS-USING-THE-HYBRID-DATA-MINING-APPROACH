import h2o
import os

h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all() #clean slate, in case cluster was already running


from h2o.estimators.gbm import H2OGradientBoostingEstimator


creditcard_df = h2o.import_file(os.path.realpath("input/creditcard.csv"))


# 60% for training
# 20% for validation (hyper parameter tuning)
# 20% for final testing

#split the data as described above
train, valid, test = creditcard_df.split_frame([0.6, 0.2], seed=1234)

#Prepare predictors and response columns
creditcard_X = creditcard_df.col_names[:-1]     #last column is Class, our desired response variable
creditcard_y = creditcard_df.col_names[-1]

gbm_v1 = H2OGradientBoostingEstimator(
    model_id="gbm_creditcard_v1",
    max_hit_ratio_k = 3,
    seed=2000000
)
gbm_v1.train(creditcard_X, creditcard_y, training_frame=train, validation_frame=valid)

gbm_v1.score_history()

gbm_v1.hit_ratio_table(valid, train = FALSE, valid = FALSE, xval = FALSE)

# This default GBM is much worse than our original random forest.
#
#
# The GBM is far from converging, so there are three primary knobs to adjust to get our performance up if we want to keep a similar run time.
#
# 1: Adding trees will help. The default is 50.
# 2: Increasing the learning rate will also help. The contribution of each tree will be stronger, so the model will move further away from the overall mean.
# 3: Increasing the depth will help. This is the parameter that is the least straightforward. Tuning trees and learning rate both have direct impact that is easy to understand. Changing the depth means you are adjusting the "weakness" of each learner. Adding depth makes each tree fit the data closer.
#
# The first configuration will attack depth the most, since we've seen the random forest focus on a continuous variable (elevation) and 40-class factor (soil type) the most.
#
# Also we will take a look at how to review a model while it is running.

# ### GBM Round 2
#
# Let's do the following:
#
# 1. decrease the number of trees to speed up runtime(from default 50 to 20)
# 2. increase the learning rate (from default 0.1 to 0.2)
# 3. increase the depth (from default 5 to 10)

# In[ ]:

gbm_v2 = H2OGradientBoostingEstimator(
    ntrees=20,
    learn_rate=0.2,
    max_depth=10,
    stopping_tolerance=0.01, #10-fold increase in threshold as defined in rf_v1
    stopping_rounds=2,
    score_each_iteration=True,
    model_id="gbm_creditcard_v2",
    seed=2000000
)
gbm_v2.train(creditcard_X, creditcard_y, training_frame=train, validation_frame=valid)


# ### Live Performance Monitoring
#
# While this is running, we can actually look at the model. To do this we simply need a new connection to H2O.
#
# This Python notebook will run the model, so we need either another notebook or the web browser (or R, etc.). In this demo, we will use [Flow](http://localhost:54321) in our web browser http://localhost:54321 and the focus will be to look at model performance, since we are using Python to control H2O.

# In[ ]:

gbm_v2.hit_ratio_table(valid=True)


# This has moved us in the right direction, but still lower accuracy than the random forest.
#
# It still has yet to converge, so we can make it more aggressive.
#
# We can now add the stochastic nature of random forest into the GBM using some of the new H2O settings. This will help generalize and also provide a quicker runtime, so we can add a few more trees.

# ### GBM: Third Time is the Charm
#
# 1. Add a few trees(from 20 to 30)
# 2. Increase learning rate (to 0.3)
# 3. Use a random 70% of rows to fit each tree
# 4. Use a random 70% of columns to fit each tree

# In[ ]:

gbm_v3 = H2OGradientBoostingEstimator(
    ntrees=30,
    learn_rate=0.3,
    max_depth=10,
    sample_rate=0.7,
    col_sample_rate=0.7,
    stopping_rounds=2,
    stopping_tolerance=0.01, #10-fold increase in threshold as defined in rf_v1
    score_each_iteration=True,
    model_id="gbm_creditcard_v3",
    seed=2000000
)
gbm_v3.train(creditcard_X, creditcard_y, training_frame=train, validation_frame=valid)


# In[ ]:

gbm_v3.hit_ratio_table(valid=True)

#test set accuracy
(final_rf_predictions['predict']==test['Cover_Type']).as_data_frame(use_pandas=True).mean()


# Our final error rates are very similar between validation and test sets. This suggests that we did not overfit the validation set during our experimentation.
