"""
Created on Mon Nov 19 02:13:47 2018

There is no stratified cross validation here

@author: hrokham
"""

#%%  libraries
###############################################################################
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.utils import resample, shuffle
from sklearn.pipeline import Pipeline
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


#%% Functions
###############################################################################
class LabelNoise:
    """
    
    """
    def __init__(
        self, 
        data,
        target, 
        pipeline,
        cv,
        name="",
        shared_noise=False, 
        verbose=0,
        output_dir = "",
        balance_method = None,
        feature_selection = None,
        noise_threshold = 0.05
    ):
        
        self.name = name
        self.output_dir = output_dir
        self.shared_noise = shared_noise 
        self.verbose = verbose 
        
        self.data = data
        self.labels = self.target_dataframe(target) # pandas df
        self.classes = list(self.labels.groupby(['given_target']).groups.keys())
        self.lbl_enc = preprocessing.LabelEncoder().fit(self.classes)
        self.labels["given_target_enc"] = self.lbl_enc.transform(self.labels["given_target"])
        # self.labels["sampling_visited"] = np.nan
        self.balance_method = balance_method
        self.sampling_iters = []
        
        self.pipeline = pipeline
        self.pipeline_results = {}
        self.cv_n_repeats = None
        self.cv_n_splits = None
        self.cv_random_state = None
        self.cv = cv
        self.cv_info()
        self.feature_selection = feature_selection
        # self.feature_selection = feature_selection
        
        self.noise_threshold  = noise_threshold  
        self.pred_noise_rate = 1
        
    def target_dataframe(self, target):
        if isinstance(target, pd.DataFrame):
            try:
                target["given_target"] = target["target"]
                target["cleansed_target"] = np.nan
                return target
            except KeyError:
                print("'target' column not found in pandas DataFrame!\n" +
                      "target DataFrame should have 'target' column.")
                
        else:
            return pd.DataFrame(data={"given_target": target,
                                      "cleansed_target": np.nan})
        
    def cv_info(self):
        if isinstance(self.cv, sklearn.model_selection._split.RepeatedStratifiedKFold):
            self.cv_n_repeats = self.cv.__dict__["n_repeats"]
            self.cv_n_splits = self.cv.__dict__["cvargs"]["n_splits"]
            self.cv_random_state = self.cv.__dict__["random_state"]
    
    def class_samplesize(self, df=None, col=['given_target']):
        """
        group by dataframe based on 'given_target' column or given columns.

        Parameters
        ----------
        df : Pandas.DataFrame, optional, default is None.
            It can be instance label dataframe or any other dataframe
        col : list of string, optional, default ['given_target']
            The column or columns want to groupby and get their size.

        Returns
        -------
        gb_size: the size of pandas groupby perfomed on dataframe using col

        """
        
        if isinstance(df, pd.DataFrame):
            gb_size = df.groupby(col).size()
            
        else:
            gb_size = self.labels.groupby(col).size()
            
        return gb_size
        
    
    def balance_sampling(self):
        """ balance sampling 
        
        the method create new columns in labels dataframe based on 
        sampling methods to handle unbalaced dataset. The function use 
        instance balance_method.
        
        Parameters
        ----------
        self.balance_method : str, default = None
            balance_method can be None, undersampling, oversampling
            
            None: No action will perfom on data regarding sampling and it only
                keep copy of given_labels into 'sampling_iter1' column
            
            undersampling: it creates multiple subset of data such that in 
            each subset the classes have equal sample size. the process 
            continues till all the samples place in at least one subset.
            
            oversampling: Not implemented yet

        Returns
        -------
        None.
            It only update labels dataframe with new columns

        """
        if self.balance_method == None:
            print("No balance sampling method applied")
            self.class_samplesize()
            self.labels["sampling_visited"] = 1
            self.labels["sampling_visited_count"] = 1
            self.labels["sampling_iter1"] = 1
            self.sampling_iters = ["sampling_iter1"]
            self.class_samplesize()
            
        elif self.balance_method == "undersampling":
            print(f"{self.balance_method} applied on data")
            self.labels["sampling_visited"] = np.nan
            self.labels["sampling_visited_count"] = 0
            i = 1
            self.sampling_iters = []
            while self.labels["sampling_visited"].isnull().any():
                iter_str = f"sampling_iter{i}"
                self.sampling_iters.append(iter_str)
                self.labels[iter_str] = np.nan
                tinds = []
                for j,c in enumerate(self.class_samplesize().keys()):
                    # print(c)
                    unseen_samples = self.labels.loc[(self.labels["given_target"] == c) & 
                                                   (self.labels["sampling_visited"].isnull())].shape[0]
                    if (unseen_samples >= self.class_samplesize().min()):
                        tind = resample(self.labels.loc[(self.labels["given_target"] == c) & 
                                                       (self.labels["sampling_visited"].isnull())], 
                                        n_samples = self.class_samplesize().min(), 
                                        replace = False).index
                        self.labels.loc[tind, "sampling_visited"] = 1
                        self.labels.loc[tind, "sampling_visited_count"] += 1
                        self.labels.loc[tind, iter_str] = 1
                        if self.verbose >= 20:
                            print(f"class {c} visited: {0}, unvisited: {tind.shape}")
                    else:
                        tind1 = resample(self.labels.loc[(self.labels["given_target"] == c) & 
                                                        (self.labels["sampling_visited"].isnull())], 
                                         n_samples = unseen_samples, 
                                         replace = False).index
                        tind2 = resample(self.labels.loc[(self.labels["given_target"] == c) & 
                                                        (self.labels["sampling_visited"].notnull())], 
                                         n_samples = (self.class_samplesize().min() - unseen_samples), 
                                         replace = False).index
                        if self.verbose >= 20:
                            print(f"class {c} visited: {tind1.shape[0]}, unvisited: {tind2.shape}")
                        # tind = pd.concat([tind1,tind2])
                        tind = np.hstack((tind1,tind2))
                        self.labels.loc[tind, "sampling_visited"] = 1
                        self.labels.loc[tind, "sampling_visited_count"] += 1
                        self.labels.loc[tind, iter_str] = 1
                
                if self.verbose >= 20:
                    print(self.class_samplesize(df=self.labels.loc[self.labels[iter_str].notnull()], 
                                                col=["given_target"]))
                i += 1
                    
            
        # elif self.balance_method == "oversampling":
        #     print(f"{self.balance_method} method not defined!")
        #     print("No balance sampling method applied")
            
        else:
             print(f"{self.balance_method} sampling method not defined!")       
        
    def feature_select(self):
        print(f"feature_select func using '{self.feature_selection}'")               
        return self.feature_selection.fit_transform(self.data, self.tables_df["given_target"])
    
    def datacleansing(self):
        self.labels["cleansed_target"] = self.labels["given_target"]
        keep_it_converge = True
        converge_iter = 0
        while keep_it_converge:
            converge_iter = converge_iter + 1

            if self.verbose >= 1:
                # print("============================================================")
                print(f"/n/n/n############## converge_iter:{converge_iter} ##############")
                # print("============================================================")

            #### step1 balance sampling 
            self.balance_sampling()
            if self.verbose >= 2:
                print("="*40)
                
            #### step2 CV classifications pipeline
            pipeline = sklearn.base.clone(self.pipeline)
            self.pipeline_results["pipeline"] = pipeline
            self.pipeline_results["pred_iter_cols"] = []
            self.pipeline_results["pred_iter_cols_correct"] = []
            for sampling_iter in self.sampling_iters:
                sampling_iter_df = self.labels.loc[self.labels[sampling_iter] == 1]
                if self.verbose >= 2:
                    print(f"\n\n__________ {sampling_iter} __________")
                elif self.verbose >= 10:
                    print(f"__________ {sampling_iter}: {sampling_iter_df.shape} __________")                    
                if self.verbose >= 2:
                    noise_cm = confusion_matrix(self.labels['given_target'], 
                                          self.labels['cleansed_target'], 
                                          labels=self.classes)
                    print("\nlabel noise confusion matrix:")
                    print("Row: given label, column: cleansed label")
                    # print(noise_cm )
                    noise_cm_df = pd.DataFrame(noise_cm)
                    print(noise_cm_df )
                    print("")
                
                
                reapnum = 1
                foldnum = 1
                sampling_iter_p_scores = []
                pred_iter_col = f"{sampling_iter}_repeat{reapnum}"
                pred_iter_col_correct = f"{sampling_iter}_repeat{reapnum}_correct"
                self.pipeline_results["pred_iter_cols"].append(pred_iter_col)
                self.pipeline_results["pred_iter_cols_correct"].append(pred_iter_col_correct)
                self.labels[pred_iter_col] = np.nan
                self.labels[pred_iter_col_correct] = np.nan
                for train_index, test_index in self.cv.split(sampling_iter_df["cleansed_target"], sampling_iter_df["cleansed_target"]):
                    if self.verbose >= 20:
                        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
                        print(train_index.max(),test_index.max())
                        print(sampling_iter_df.iloc[train_index].index.max(),sampling_iter_df.iloc[test_index].index.max())
                    train_index = sampling_iter_df.iloc[train_index].index.values 
                    test_index = sampling_iter_df.iloc[test_index].index.values
                    
                    # train and predict
                    pipeline = sklearn.base.clone(self.pipeline)
                    pipeline.fit(self.data[train_index],
                                 self.labels.loc[train_index, 
                                                    "cleansed_target"]) 
                    
                    p_predict = pipeline.predict(self.data[test_index] )
                    self.labels.loc[test_index, pred_iter_col] = p_predict
                    
                    correct_chk = self.labels.loc[test_index, "cleansed_target"] == p_predict
                    self.labels.loc[test_index, pred_iter_col_correct] = correct_chk 
                    
                    p_score = pipeline.score(self.data[test_index], 
                                             self.labels.loc[test_index, 
                                                                "cleansed_target"])
                    
                    sampling_iter_p_scores.append(p_score)
                    
                    if self.verbose >= 1:
                        print(f"model score: {p_score:.3f}") 
                        
                    if foldnum < self.cv_n_splits:
                        foldnum += 1 
                    elif reapnum < self.cv_n_repeats:
                        reapnum += 1
                        foldnum = 1
                        pred_iter_col = f"{sampling_iter}_repeat{reapnum}"
                        pred_iter_col_correct = f"{sampling_iter}_repeat{reapnum}_correct"
                        self.pipeline_results["pred_iter_cols"].append(pred_iter_col)
                        self.pipeline_results["pred_iter_cols_correct"].append(pred_iter_col_correct)
                        self.labels[pred_iter_col] = np.nan
                        self.labels[pred_iter_col_correct] = np.nan
                    
                self.pipeline_results[f"{sampling_iter}_cv_scores"] = sampling_iter_p_scores
                
            #### step4 voting
            cols_correct = self.pipeline_results["pred_iter_cols_correct"]
            df_sum = self.labels[cols_correct].sum(axis=1, min_count=1) 
            correct_counts_str = f"correct_count"
            self.labels[correct_counts_str] = df_sum / self.labels["sampling_visited_count"]
            noise_str = f"noise"
            self.labels[noise_str] = self.labels[correct_counts_str] < (self.cv_n_repeats*0.5)
            
            self.pred_noise_rate = self.labels[noise_str].sum() / self.labels.shape[0]
            if self.verbose >= 1:
                print(f"current noise rate: {self.pred_noise_rate:.4f}")
            
            #### step5 remove label noise
            if self.pred_noise_rate  < self.noise_threshold :
                keep_it_converge = False
                X_train, X_test, y_train, y_test = train_test_split(self.data, 
                                                                    self.labels["cleansed_target"], 
                                                                    test_size=0.2,
                                                                    # random_state=1, 
                                                                    stratify=self.labels["cleansed_target"])
                pipeline = sklearn.base.clone(self.pipeline)
                pipeline.fit(X_train, y_train) 
                print('Cleansed Test Accuracy: %0.3f' % pipeline.score(X_test, y_test))
            else:
                noisy_ind = self.labels.loc[self.labels[noise_str]].index.values
                cleansed_ind = self.labels.loc[~self.labels[noise_str]].index.values
                X_cleansed = self.data[cleansed_ind]
                y_cleansed = self.labels.loc[cleansed_ind, "cleansed_target"]
                pipeline = sklearn.base.clone(pipeline)
                pipeline.fit(X_cleansed, y_cleansed) 
                new_labels = pipeline.predict(self.data[noisy_ind] )
                self.labels.loc[noisy_ind, "cleansed_target"] = new_labels
                        
        return self.labels.loc[noisy_ind, "cleansed_target"]
