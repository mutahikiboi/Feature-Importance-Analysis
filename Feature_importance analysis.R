install.packages("bbotk")
packageVersion("bbotk")
install.packages("mlr3verse")
library(bbotk)
library(mlr3verse)
library(mlr3filters)
library(mlr3fselect)
library(mlr3tuning)
library(mlr3pipelines)
library(tidyverse)

#defining out task

task <- tsk("spam")
task
#all features are numeric
task %>% view
task$data() 
task%>% summary
task$missings()

length(task)

#plotting the task
task %>% autoplot()

task$missings()
#no missing values

mlr_tuners %>% as.data.table() %>% view()
#as.data.table(mlr_tuners) %>% view
#We have only a slight class imbalance, and there are no missing values. Also, all features are
#numeric

#learner1 = lrn("classif.rpart")

#learner1$param_set

#view feature filters belonging to the mlr3 package
mlr_filters %>% 
as.data.table() %>%
.[packages == "mlr3"]%>%
view()

#creating an importance filter for decision tree
#load library for decision trees
library(rpart)
set.seed(42)
lrn.rpart = lrn("classif.rpart") #importance parameter cannot be set directly in decision tree
filter = flt("importance", learner = lrn.rpart)
filter$calculate(task)
filter$scores
as.data.table(filter) #orders the scores in descending order

#creating an importance filter for random forest
set.seed(42) # for reproducibility
lrn.ranger = lrn("classif.ranger", importance = "impurity")
filter = flt("importance", learner = lrn.ranger)
filter$calculate(task)
filter$scores

as.data.table(filter) #orders the scores in descending order

?ranger::ranger

#c
#Integrating the importance filter with the random forest into a GraphLearner
#as a pre-processing step for a random forest learner.
#then optimize the number of features chosen w.r.t. the performance as indicated by the AUC measure (msr("classif.auc")).
graph_ranger <- as_learner( po("filter", flt( "importance",learner = lrn.ranger)) %>>%
    lrn("classif.ranger", predict_type = "prob")
    )
#--------------------------------------------------------------------------------------------------------------------------
#same as above 
#ranger_filter <- po("filter",flt("importance", learner = lrn.ranger) ) %>% lrn("classif.ranger", predict_type = "prob")
#graph_ranger <- as_learner(ranger_filter)
#--------------------------------------------------------------------------------------------------------------------------
#The GraphLearner combines feature selection and model training into a single pipeline.
#failure to do this means the autotuner has to search through all the 57 features which is not optimal as it does not know which feature are relevant

tuner_nfeat <- mlr3tuning::AutoTuner$new(
  learner = graph_ranger,
  resampling = rsmp("holdout"),
  tuner = tnr("grid_search", resolution =10),
  measure = msr("classif.auc"),
  terminator = trm("none"),
  search_space = ps(
    importance.filter.nfeat = p_int(1, 57, logscale =FALSE)
  )
)
tuner_nfeat$train(task)
#when we review the scores we do not get any meaningful insight and we need to plot them scores
tuner_nfeat$tuning_instance

#Plotting the feature importance based on auc
--------------------------------------------------------------------------------------------------
tuner_nfeat$tuning_instance %>%
  autoplot() #should return the same as below

#or

autoplot(tuner_nfeat$tuning_instance) #this is how you plot
------------------------------------------------------------------------------------------------
tuner_nfeat$tuning_instance %>%
  autoplot(cols_x = c("importance.filter.nfeat"))

#d Sequential wrapper with a decision tree learner.
#with stagnation terminator (with iters = 100) then we evaluate the performance as before
library(mlr3fselect)
set.seed(42)
resampling = rsmp("cv", folds = 10L)
resampling$instantiate(task)

learner = lrn("classif.rpart", predict_type = "prob")
meas = msr("classif.auc")

#this is a wrapper
instance = FSelectInstanceBatchSingleCrit$new(
    task = task,
    learner = learner,
    resampling = resampling,
    measure = meas,
    terminator = trm("stagnation", iters =10),
    store_models = TRUE, #as models grow bigger you cannot store
)
     
fselector <- fs("sequential")

#we optimize the instance   
fselector$optimize(instance)

#Here, we would select the following seven features:
instance$result_feature_set
    
as.data.table(instance$archive) %>%
as_tibble()

as.data.table(instance$archive) %>%
view()

#we are just creating an autoFselector
auto_fs = AutoFSelector$new(
  learner = learner,
  resampling = rsmp("holdout"), #inner resampling
  measure = msr("classif.auc"),
  terminator = trm("stagnation", iters =10), #100
  fselector = fs("sequential"),
  store_models = TRUE
)

#Benchmark the filter and wrapper approaches, use holdout for inner resampling, and cv for outer resampling 
#benchmark also with a baseline approach
learners <- list(auto_fs, 
                 tuner_nfeat, 
                 lrn("classif.rpart", predict_type = "prob")# baseline
                 )

r_cv <- rsmp("cv") #outer resampling

design <- benchmark_grid(
  task=task,
  resampling =r_cv,
  learners = learners,
  )
bmr <-benchmark(design) 


#bmr <- benchmark(benchmark_grid(task, learners, r_cv))

#Measure <- msrs(msrs("classif.auc", "classif.bacc"))
bmr$aggregate(msrs("classif.auc"))
autoplot(bmr)

#2.  mlr_fselectors object and choose a feature selection method

task2 <- tsk("sonar")

#Create two AutoFSelector objects, genetic search and random_search

inner_resampling <- rsmp("cv", folds = 3)
measure <- msr("classif.bacc")
terminator <- trm("evals", n_evals = 100)

afs_random <- AutoFSelector$new(
  learner = lrn("classif.kknn"),
  resampling = inner_resampling,
  measure = measure,
  terminator = terminator,
  fselector = fs("random_search")
)
afs_random$id <- "random_fs"

afs_genetic <- AutoFSelector$new(
                                  learner = lrn('classif.kknn'),
                                  resampling = inner_resampling,
                                  measure = measure,
                                  terminator = terminator,
                                   fselector = fs("genetic_search",
                                                 popSize = 10L,
                                                 elitism = 2L,
                                                zeroToOneRatio = 2L)
                                )

afs_genetic$id <- "genetic_fs"


outer_resampling <- rsmp('cv',folds = 3)

learners2 <- list(
                  afs_random,
                  afs_genetic,
                  lrn('classif.kknn'),
                  lrn('classif.featureless')
                  )

design <- benchmark_grid(tasks = task2,
                        learners = learners2,
                        resamplings = outer_resampling
                        )
bmr <- benchmark(design = design, store_models = TRUE)
bmr %>% autoplot()
  
  
  
  