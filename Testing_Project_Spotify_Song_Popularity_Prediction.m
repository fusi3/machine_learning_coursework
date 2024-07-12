%% Loading the final trained models and the testing data

% The models

loaded_model_1 = load('final_base_logistic_classifier.mat');
loaded_model_2 = load('final_tuned_logistic_classifier.mat');
loaded_model_3 = load('final_base_random_forest_classifier.mat');
loaded_model_4 = load('final_tuned_random_forest_classifier.mat');

final_base_logistic_classifier = loaded_model_1.final_base_logistic_classifier;
final_tuned_logistic_classifier = loaded_model_2.final_tuned_logistic_classifier;
final_base_random_forest_classifier = loaded_model_3.final_base_random_forest_classifier;
final_tuned_random_forest_classifier = loaded_model_4.final_tuned_random_forest_classifier;

% The testing data

test_data_lr = readtable('test_data_lr.csv');
test_data_rf = readtable('test_data_rf.csv');

% The data to scale the logistic regression testing data

training_iqr_values = readtable('training_iqr_values.csv');
training_medians = readtable('training_medians.csv');

%% Data preparation for the logistic models
%we are now going to prepare the data like we did for the training
%examples. Scale for logistic and get the labels


%for logistic regression classifier we are going to scale the data with the
%robust scaling terms from the train. Remember we partitioned our data and
%the test we are using here was not processed in the training script. This
%was done to avoid data bleed/leakage when scaling so that the test set is
%being used completely blind by the model

%Documentation
%https://arxiv.org/pdf/2108.02497.pdf - page 5 - train test leaks

final_testing_labels_lr = test_data_lr(:, 10);
test_data_lr(:, 10) = [];
final_testing_data_lr = test_data_lr;

final_testing_data_lr_numericals = final_testing_data_lr(:, 1:10);

repeated_training_medians = repmat(training_medians, size(final_testing_data_lr_numericals, 1), 1); 
repeated_training_iqr_values = repmat(training_iqr_values, size(final_testing_data_lr_numericals, 1), 1);

training_iqr_truncated_names = extractAfter(repeated_training_iqr_values.Properties.VariableNames, 4);
repeated_training_iqr_values = renamevars(repeated_training_iqr_values, ...
    repeated_training_iqr_values.Properties.VariableNames, ...
    training_iqr_truncated_names);

% Robust scaling
robust_scaled_final_testing_data_lr_numericals = ...
(final_testing_data_lr_numericals - repeated_training_medians) ./ repeated_training_iqr_values;

final_testing_data_dummies = final_testing_data_lr(:, 11:end);
final_testing_data_lr = [robust_scaled_final_testing_data_lr_numericals final_testing_data_dummies];

final_testing_data_lr_array = table2array(final_testing_data_lr);
final_testing_labels_lr_array = table2array(final_testing_labels_lr);

%% Base Logistic Model

% setting up the base logistic model and calculating the metrics

base_lr_predicted_probabilities = predict(final_base_logistic_classifier, final_testing_data_lr_array);

base_lr_predicted_classes = base_lr_predicted_probabilities > 0.5;

TP_base_lr = sum(base_lr_predicted_classes & final_testing_labels_lr_array); 
FN_base_lr = sum(~base_lr_predicted_classes & final_testing_labels_lr_array); 
FP_base_lr = sum(base_lr_predicted_classes & ~final_testing_labels_lr_array); 
TN_base_lr = sum(~base_lr_predicted_classes & ~final_testing_labels_lr_array); 

%metric calculations

precision_base_lr = TP_base_lr / (TP_base_lr + FP_base_lr);
recall_base_lr = TP_base_lr / (TP_base_lr + FN_base_lr);
accuracy_base_lr = (TP_base_lr + TN_base_lr) / (TP_base_lr + TN_base_lr + FP_base_lr + FN_base_lr);
F1_base_lr = 2 * (precision_base_lr * recall_base_lr) ...
/ (precision_base_lr + recall_base_lr); 

[false_positive_rate, true_positive_rate, thresholds, auc_base_lr] = ...
perfcurve(final_testing_labels_lr_array, base_lr_predicted_probabilities, true); %#ok<*ASGLU>

tested_base_logistic_classifier_metrics = table();

tested_base_logistic_classifier_metrics.Precision = precision_base_lr;
tested_base_logistic_classifier_metrics.Accuracy = accuracy_base_lr;
tested_base_logistic_classifier_metrics.Recall = recall_base_lr;
tested_base_logistic_classifier_metrics.F1 = F1_base_lr;
tested_base_logistic_classifier_metrics.AUC = auc_base_lr;

writetable(tested_base_logistic_classifier_metrics, ...
    'tested_base_logistic_classifier_metrics.csv');

% plotting the ROC curve

tested_base_logistic_ROC = figure('Position', [100, 100, 800, 800]);
plot(false_positive_rate, true_positive_rate)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Tested Base Logistic Classifier ROC Curve')
saveas(tested_base_logistic_ROC, 'Tested Base Logistic Classifier ROC Curve.png');

%curve not looking good. Base model not preforming well on test


% creating a confusion matrix
tested_base_logistic_classifier_confusion_matrix = [TP_base_lr, FN_base_lr; FP_base_lr, TN_base_lr]; 
tested_base_logistic_classifier_confusion_matrix_normalized = ...
100* ...
(tested_base_logistic_classifier_confusion_matrix ...
/ sum(tested_base_logistic_classifier_confusion_matrix(:)));

tested_base_logistic_confusion_matrix = figure('Position', [100, 100, 800, 800]);
heatmap(tested_base_logistic_classifier_confusion_matrix_normalized, 'XDisplayLabels', {1, 0}, ...
    'YDisplayLabels', {1, 0});
xlabel('Predicted');    
ylabel('Actual');
title('Tested Base Logistic Confusion Matrix Percents');
saveas(tested_base_logistic_confusion_matrix, 'Tested Base Logistic Confusion Matrix Percents.png');

%bad preformance validatd by confusion matrix

%% Tuned Logistic Model

% setting up the tuned logistic model and calculating the metrics

tuned_lr_predicted_probabilities = predict(final_tuned_logistic_classifier, final_testing_data_lr_array);

tuned_lr_predicted_classes = tuned_lr_predicted_probabilities > 0.5;

TP_tuned_lr = sum(tuned_lr_predicted_classes & final_testing_labels_lr_array); 
FN_tuned_lr = sum(~tuned_lr_predicted_classes & final_testing_labels_lr_array); 
FP_tuned_lr = sum(tuned_lr_predicted_classes & ~final_testing_labels_lr_array); 
TN_tuned_lr = sum(~tuned_lr_predicted_classes & ~final_testing_labels_lr_array); 

%metric calculations

precision_tuned_lr = TP_tuned_lr / (TP_tuned_lr + FP_tuned_lr);
recall_tuned_lr = TP_tuned_lr / (TP_tuned_lr + FN_tuned_lr);
accuracy_tuned_lr = (TP_tuned_lr + TN_tuned_lr) / (TP_tuned_lr + TN_tuned_lr + FP_tuned_lr + FN_tuned_lr);
F1_tuned_lr = 2 * (precision_tuned_lr * recall_tuned_lr) ...
/ (precision_tuned_lr + recall_tuned_lr); 

[false_positive_rate, true_positive_rate, thresholds, auc_tuned_lr] = ...
perfcurve(final_testing_labels_lr_array, tuned_lr_predicted_probabilities, true);

tested_tuned_logistic_classifier_metrics = table();

tested_tuned_logistic_classifier_metrics.Precision = precision_tuned_lr;
tested_tuned_logistic_classifier_metrics.Accuracy = accuracy_tuned_lr;
tested_tuned_logistic_classifier_metrics.Recall = recall_tuned_lr;
tested_tuned_logistic_classifier_metrics.F1 = F1_tuned_lr;
tested_tuned_logistic_classifier_metrics.AUC = auc_tuned_lr;

writetable(tested_tuned_logistic_classifier_metrics, ...
    'tested_tuned_logistic_classifier_metrics.csv');

% plotting the ROC curve

tested_tuned_logistic_ROC = figure('Position', [100, 100, 800, 800]);
plot(false_positive_rate, true_positive_rate)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Tested Tuned Logistic Classifier ROC Curve')
saveas(tested_tuned_logistic_ROC, 'Tested Tuned Logistic Classifier ROC Curve.png');

%curve not looking good. Base model not preforming well on test

% creating a confusion matrix
tested_tuned_logistic_classifier_confusion_matrix = [TP_tuned_lr, FN_tuned_lr; FP_tuned_lr, TN_tuned_lr]; 
tested_tuned_logistic_classifier_confusion_matrix_normalized = ...
100* ...
(tested_tuned_logistic_classifier_confusion_matrix ...
/ sum(tested_tuned_logistic_classifier_confusion_matrix(:)));

tested_tuned_logistic_confusion_matrix = figure('Position', [100, 100, 800, 800]);
heatmap(tested_tuned_logistic_classifier_confusion_matrix_normalized, 'XDisplayLabels', {1, 0}, ...
    'YDisplayLabels', {1, 0});
xlabel('Predicted');    
ylabel('Actual');
title('Tested Tuned Logistic Confusion Matrix Percents');
saveas(tested_tuned_logistic_confusion_matrix, 'Tested Tuned Logistic Confusion Matrix Percents.png');

%bad preformance validated by confusion matrix. tuning has not made a
%difference on the test set

%% Random Forest Data Preparation

final_testing_labels_rf = test_data_rf(:, 13);
test_data_rf(:, 13) = [];
final_testing_data_rf = test_data_rf;
final_testing_data_rf = convertvars(final_testing_data_rf,{'key','mode','time_signature'},'categorical');

final_testing_labels_rf_array = table2array(final_testing_labels_rf);

%% Base Random Forest

% setting up the tuned logistic model and calculating the metrics

base_rf_predicted_probabilities = predict(final_base_random_forest_classifier, final_testing_data_rf);

base_rf_predicted_probabilities_numeric = cellfun(@(x) double(strcmp(x, '1')), base_rf_predicted_probabilities);

TP_base_rf = sum(base_rf_predicted_probabilities_numeric & final_testing_labels_rf_array); 
FN_base_rf = sum(~base_rf_predicted_probabilities_numeric & final_testing_labels_rf_array); 
FP_base_rf = sum(base_rf_predicted_probabilities_numeric & ~final_testing_labels_rf_array); 
TN_base_rf = sum(~base_rf_predicted_probabilities_numeric & ~final_testing_labels_rf_array); 

%metric calculations

precision_base_rf = TP_base_rf / (TP_base_rf + FP_base_rf);
recall_base_rf = TP_base_rf / (TP_base_rf + FN_base_rf);
accuracy_base_rf = (TP_base_rf + TN_base_rf) / (TP_base_rf + TN_base_rf + FP_base_rf + FN_base_rf);
F1_base_rf = 2 * (precision_base_rf * recall_base_rf) ...
/ (precision_base_rf + recall_base_rf); 

[false_positive_rate, true_positive_rate, thresholds, auc_base_rf] = ...
perfcurve(final_testing_labels_rf_array, base_rf_predicted_probabilities_numeric, true);

tested_base_random_foresr_classifier_metrics = table();

tested_base_random_foresr_classifier_metrics.Precision = precision_base_rf;
tested_base_random_foresr_classifier_metrics.Accuracy = accuracy_base_rf;
tested_base_random_foresr_classifier_metrics.Recall = recall_base_rf;
tested_base_random_foresr_classifier_metrics.F1 = F1_base_rf;
tested_base_random_foresr_classifier_metrics.AUC = auc_base_rf;

writetable(tested_base_random_foresr_classifier_metrics, ...
    'tested_base_random_forest_classifier_metrics.csv');

% plotting the ROC curve

tested_base_random_forest_ROC = figure('Position', [100, 100, 800, 800]);
plot(false_positive_rate, true_positive_rate)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Tested Base Random Forest Classifier ROC Curve')
saveas(tested_base_random_forest_ROC, 'Tested Base Random Forest Classifier ROC Curve.png');

%curve looking a bit better than the logistic curves

% creating a confusion matrix
tested_base_random_forest_classifier_confusion_matrix = [TP_base_rf, FN_base_rf; FP_base_rf, TN_base_rf]; 
tested_base_random_forest_classifier_confusion_matrix_normalizd = ...
100* ...
(tested_base_random_forest_classifier_confusion_matrix ...
/ sum(tested_base_random_forest_classifier_confusion_matrix(:)));

tested_base_random_forest_confusion_matrix = figure('Position', [100, 100, 800, 800]);
heatmap(tested_base_random_forest_classifier_confusion_matrix_normalizd, 'XDisplayLabels', {1, 0}, ...
    'YDisplayLabels', {1, 0});
xlabel('Predicted');    
ylabel('Actual');
title('Tested Base Random Forest Confusion Matrix Percents');
saveas(tested_base_random_forest_confusion_matrix, 'Tested Base Random Forest Confusion Matrix Percents.png');

%improved preformance on predicting the negative class but the positive
%class is still a problem. Surprisingly, the logistic was tuned for
%precision but the base random forest has a better precision

%% Tuned Random Forest

%% Base Random Forest

% setting up the tuned logistic model and calculating the metrics

tuned_rf_predicted_probabilities = predict(final_tuned_random_forest_classifier, final_testing_data_rf);

tuned_rf_predicted_probabilities_numeric = cellfun(@(x) double(strcmp(x, '1')), tuned_rf_predicted_probabilities);

TP_tuned_rf = sum(tuned_rf_predicted_probabilities_numeric & final_testing_labels_rf_array); 
FN_tuned_rf = sum(~tuned_rf_predicted_probabilities_numeric & final_testing_labels_rf_array); 
FP_tuned_rf = sum(tuned_rf_predicted_probabilities_numeric & ~final_testing_labels_rf_array); 
TN_tuned_rf = sum(~tuned_rf_predicted_probabilities_numeric & ~final_testing_labels_rf_array); 

%metric calculations

precision_tuned_rf = TP_tuned_rf / (TP_tuned_rf + FP_tuned_rf);
recall_tuned_rf = TP_tuned_rf / (TP_tuned_rf + FN_tuned_rf);
accuracy_tuned_rf = (TP_tuned_rf + TN_tuned_rf) / (TP_tuned_rf + TN_tuned_rf + FP_tuned_rf + FN_tuned_rf);
F1_tuned_rf = 2 * (precision_tuned_rf * recall_tuned_rf) ...
/ (precision_tuned_rf + recall_tuned_rf); 

[false_positive_rate, true_positive_rate, thresholds, auc_tuned_rf] = ...
perfcurve(final_testing_labels_rf_array, tuned_rf_predicted_probabilities_numeric, true);

tested_tuned_random_foresr_classifier_metrics = table();

tested_tuned_random_foresr_classifier_metrics.Precision = precision_tuned_rf;
tested_tuned_random_foresr_classifier_metrics.Accuracy = accuracy_tuned_rf;
tested_tuned_random_foresr_classifier_metrics.Recall = recall_tuned_rf;
tested_tuned_random_foresr_classifier_metrics.F1 = F1_tuned_rf;
tested_tuned_random_foresr_classifier_metrics.AUC = auc_tuned_rf;

writetable(tested_tuned_random_foresr_classifier_metrics, ...
    'tested_tuned_random_forest_classifier_metrics.csv');

% plotting the ROC curve

tested_tuned_random_forest_ROC = figure('Position', [100, 100, 800, 800]);
plot(false_positive_rate, true_positive_rate)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Tested Tuned Random Forest Classifier ROC Curve')
saveas(tested_tuned_random_forest_ROC, 'Tested Tuned Random Forest Classifier ROC Curve.png');

%curve not looking good. Base model not preforming well on test

% creating a confusion matrix
tested_tuned_random_forest_classifier_confusion_matrix = [TP_tuned_rf, FN_tuned_rf; FP_tuned_rf, TN_tuned_rf]; 
tested_tund_random_forest_classifier_confusion_matrix_normalizd = ...
100* ...
(tested_tuned_random_forest_classifier_confusion_matrix ...
/ sum(tested_tuned_random_forest_classifier_confusion_matrix(:)));

tested_tuned_random_forest_confusion_matrix = figure('Position', [100, 100, 800, 800]);
heatmap(tested_tund_random_forest_classifier_confusion_matrix_normalizd, 'XDisplayLabels', {1, 0}, ...
    'YDisplayLabels', {1, 0});
xlabel('Predicted');    
ylabel('Actual');
title('Tested Tuned Random Forest Confusion Matrix Percents');
saveas(tested_tuned_random_forest_confusion_matrix, 'Tested Tuned Random Forest Confusion Matrix Percents.png');

%% ---------- CODE IS GOOD TILL HERE -----------------
