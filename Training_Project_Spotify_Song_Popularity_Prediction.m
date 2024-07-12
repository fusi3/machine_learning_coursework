%% Loading the data in
%we will specify the path of the file and load the csv data into a table
file_path = 'C:\Users\Fasih Munir\Desktop\Sub Desktop\Study\University\Masters - City University London 20230603\Machine Learning\Coursework\spotify_data.csv';
spotify = readtable(file_path);

%the EDA below is done on a random sample of ~10K observations.
%it did not make sense to the EDA on all data and carry forward that
%thinking into a random subset due to random variability of the split.
%Although upon observations, mose of the observations hold on the random
%subset as well
%Initially I used all 1.2 million rows
%but that results in the random forests taking too long to train (~4.5 hours for the whole script) 
%and was why I chose to leave out cross validation for the final tuned model. But
%even after that and saving the model the random forests were 2gb large
%which were not even loading fast enough to test. To save time, I have
%continued to not use cross val on the tuned random forest and am now using
%only 10k rows. These will be stratified to ensure the imbalance (described
%below) will carry forward

%% Some initial EDA
%Variable Documentation
%https://developer.spotify.com/documentation/web-api/reference/get-audio-features

%we will create our target column using a condition on the popularity. We
%are setting up a classification problem. Most problems had a regression
%problems but I wanted to try something a little different while keeping
%the concept the same. The choice to split on 20 popularity was to reduce
%the effect of the imbalance but still maintain a little imbalance to
%distinguish songs that became popular

%what is the popularity index based off? It is a spotify algorithm that
%looks at the number of streams, the recency of streams and other things
%like number of adds to a playlist etc 
% (see the documentation here
% https://community.spotify.com/t5/Content-Questions/Artist-popularity/td-p/4415259)
%that makes time an important component as songs could have been popular
%and are no longer popular because others have taken their place. For now
%we will simply remove the year field to avoid the complexity
%it does however seem to be the case that there are less than 5% of songs
%that reached a popularity of greater than or equal to 80. For this work we
%will go with a value that is 20 to improve the balance

is_popular = zeros(size(spotify.popularity)); %creating the new column initially with only zeros
is_popular(spotify.popularity >= 20) = 1; %assigning the value of 1 if the popularity score is greater than 80
spotify.is_popular = is_popular; %this adds the new column to our data
total_rows = height(spotify); %finding the count of rows in the table
unique_rows = unique(spotify.track_id); %finding the unique tracks and saving them separtely to count later
count_unique_rows = height(unique_rows); %this code did not output in the command window
%but the results can be seen in the workspace window
missing_data = sum(ismissing(spotify)); %there are no missing values

rng(666);

subset_labels_1 = spotify.is_popular;
subset_all_columns_1 = spotify.Properties.VariableNames;
subset_features_1 = spotify(:, subset_all_columns_1);
subset_holdout_percent_1 = 0.9913;
subset_cv_1 = cvpartition(subset_labels_1, 'Holdout', subset_holdout_percent_1, 'Stratify', true);

subset_train_index_1 = training(subset_cv_1);
subset_test_index_1 = test(subset_cv_1);

spotify = subset_features_1(subset_train_index_1, :);
remaining_spotify_data = subset_features_1(subset_test_index_1, :);

popular_counts = countcats(categorical(spotify.is_popular)); % Counts the occurrences of each unique value
distinct_values = unique(spotify.is_popular); %this will help us use the values as the labels of the bars
% we will now plot a bar chart to see how much imbalance in the target
% variable we have
target_class_ratio = figure('Position', [100, 100, 800, 800]);
bar(distinct_values, popular_counts)
xlabel('Is Popular')
ylabel('Counts')
title('Count of Popular Songs (1) vs Not Popular Songs (0)')
% Displaying total counts on top of each bar - chatgpt
text(distinct_values, popular_counts, arrayfun(@num2str, popular_counts, 'UniformOutput', false), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom')
saveas(target_class_ratio, 'target_class_ratio.png');

%lets check the distributions of the numeric variables

%Documentation
%https://uk.mathworks.com/help/matlab/ref/for.html
%https://uk.mathworks.com/help/matlab/ref/subplot.html
%https://uk.mathworks.com/help/matlab/ref/sgtitle.html
%and debugging with chatgpt

%year has been removed (described above) as it has a difficult
%relationship with popularity given how popularity is calculated so to keep
%things simple have removed it
numeric_variables = {'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', ...
    'liveness', 'valence', 'tempo', 'duration_ms'}; %data is in a table so need to run this code else it would show as a cell which causes problem 
distribution_plot_numeric = figure('Position', [100, 100, 1000, 800]);
for i = 1:width(spotify(:, numeric_variables))
    subplot(3, 4, i);
    histogram(spotify.(numeric_variables{i}), 20);
    title(numeric_variables{i});
end
sgtitle('Distributions of Numeric Variables');
saveas(distribution_plot_numeric, 'distribution_plot_numeric.png');
%most of the variables do not follow a normal distribution and are quite
%skewed towards one direction. Likely can not use z score normalization


%lets check out the summary stats of the numeric variables
numeric_data = spotify(:, {'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', ...
    'liveness', 'valence', 'tempo', 'duration_ms'}); %data is in a table so need to run this code else it would show as a cell which causes problem 

summary_mean = table2cell(mean(numeric_data));
summary_std = table2cell(std(numeric_data));
summary_mode = table2cell(mode(numeric_data));
summary_median = table2cell(median(numeric_data));
summary_min = table2cell(min(numeric_data));
summary_max = table2cell(max(numeric_data));

stats_summary_table = array2table([summary_mean; summary_std; summary_mode; summary_median; ... %apostrophy will transpose the table
    summary_min; summary_max], ...
    'VariableNames', numeric_variables, ...
    'RowNames', {'Mean', 'Std', 'Mode', 'Median', 'Min', 'Max'});
writetable(stats_summary_table, 'stats_summary_table.csv');

%lets check the distribution of the non numeric variables
non_numeric_variables = {'key', 'mode', 'time_signature'};
distribution_plot_non_numeric = figure('Position', [100, 100, 1000, 800]);
for i = 1:width(spotify(:, non_numeric_variables))
    subplot(3, 1, i);
    histogram(categorical(spotify.(non_numeric_variables{i})), 'BarWidth', 1);
    title(non_numeric_variables{i});
end
sgtitle('Distributions of Non Numeric Variables')
saveas(distribution_plot_non_numeric, 'distribution_plot_non_numeric.png');


%lets plot a spearman correlation map to get a better idea about how the
%numeric variables are related
%we include popularity which is not our final target but will give us an idea about
%the variables that will have an impact
%using spearman correlation (instead of pearson) as the variables are not normally distributed - see distribution_plot_numeric

%Documentation
%https://uk.mathworks.com/help/matlab/ref/corrcoef.html
%https://uk.mathworks.com/help/matlab/ref/colorbar.html
%https://uk.mathworks.com/help/matlab/ref/imagesc.html
%https://uk.mathworks.com/matlabcentral/answers/496872-displaying-matrix-of-correlation-coefficients?s_tid=prof_contriblnk
%https://journals.lww.com/anesthesia-analgesia/fulltext/2018/05000/correlation_coefficients__appropriate_use_and.50.aspx
%- when to use what correlation
corr_data_matrix = table2array(numeric_data);
spearman_corr_data_matrix = corr(corr_data_matrix, 'type','Spearman');
corr_plot_numeric = figure('Position', [100, 100, 800, 800]);
imagesc(spearman_corr_data_matrix);
colorbar;
title('Spearman Correlation Plot For Numeric Variables');
xlabel('Variables');
ylabel('Variables');
xticks(1:size(spearman_corr_data_matrix, 1)); %chatgpt
yticks(1:size(spearman_corr_data_matrix, 1));
xticklabels(numeric_data.Properties.VariableNames);
yticklabels(numeric_data.Properties.VariableNames);
saveas(corr_plot_numeric, 'corr_plot_numeric.png');

%variables that have correlation intuitvely seem expected but it is not
%extremely high. Most other variables have low correlation which is good
%for the models

%% Preprocessing for logistic regression classifier

%lets now build a data set that is ready to be used for machine learning by
%removing and scaling the columns

logistic_spotify = spotify; %here i am making a copy of the original table and this is what we will work with
logistic_spotify.duration_minutes = round(logistic_spotify.duration_ms / 60000,1); %convert milliseconds to minutes and round to 1 decimal place

%qqplot(logistic_spotify.duration_minutes); %this is not really normally distributed so can not use z score noramlisation
%qqplot(logistic_spotify.tempo); %this is mostly normally distributed so can use z score normalisation
outliers_minutes = figure('Position', [100, 100, 1000, 800]);
boxplot(logistic_spotify.duration_minutes);
title('Boxplot For Duration Minutes');
saveas(outliers_minutes, 'outliers_minutes.png');
%there are a lot of outliers in the duration minutes which makes us choose
%between min max scaling or robust scaling
%we could scale different variables separately and this may affect the
%outcomes but for now we will use robust scaling to deal with the outliers
%and apply that to all the variables

%we are now going to make dummy variables on specific columns and then will
%append these back to our useable table
%it should be noted that spotifys api has already changed the strings to
%numerics example the key of D is mapped to 2. However, since the order
%does not matter, even the numerical mapping can be treated the same way as
%the original string
%for each variable, we will dorp the last dummy variable produced to reduce
%the effect of multicollinearity. Gender male and Gender female would have
%perfect colinearity so best to leave one out

%Documentation
%https://uk.mathworks.com/help/stats/train-svm-classifier-with-categorical-predictors-and-generate-c-code.html
%https://www.mdpi.com/2227-7390/10/8/1283 - multicolinearity problem

key_category = categorical(logistic_spotify.key);
mode_category = categorical(logistic_spotify.mode);
time_signature_category = categorical(logistic_spotify.time_signature);

unique_key = transpose(categories(key_category));
unique_mode = transpose(categories(mode_category));
unique_time_signature = transpose(categories(time_signature_category));

dummy_key = dummyvar(key_category);
dummy_mode = dummyvar(mode_category);
dummy_time_signature = dummyvar(time_signature_category);

dummy_table_key = array2table(dummy_key, 'VariableNames', strcat('key_', cellstr(unique_key)));
dummy_table_mode = array2table(dummy_mode, 'VariableNames', strcat('mode_', cellstr(unique_mode)));
dummy_table_time_signature = array2table(dummy_time_signature, 'VariableNames', strcat('time_signature_', cellstr(unique_time_signature)));

dummy_table_time_signature(:, end) = [];
dummy_table_mode(:, end) = [];
dummy_table_key(:, end) = [];

logistic_spotify = [logistic_spotify, dummy_table_key];
logistic_spotify = [logistic_spotify, dummy_table_mode];
logistic_spotify = [logistic_spotify, dummy_table_time_signature];
logistic_spotify;

%% Creating the first split to get the test data
%we will now split the data into train and test sets using stratified cv to
%maintain the imbalance. Once it is split we will scale the train set
%using robust scaler to account for outliers (specifically for duration
%minutes). The same scaling parameters will then be used on the test
%set. min max scaler would have been used if the outliers in duration
%minutes were not so large. Standardizing with z score does not make sense
%since most variables are between 0 and 1 already and there is only 1
%variable that most likely follows a normal distribution

%Documentation
%https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10217387/ - cross validation
%improves results
%https://arxiv.org/pdf/2108.02497.pdf - page 5 - dont want data to leak
%during scaling
%https://arxiv.org/pdf/2212.12343.pdf - when to use what kind of scaling
%page 8
%https://rpubs.com/annabauer/940476 - build average artist popularity

rng(2023);  %setting default seed for reproducability
labels = logistic_spotify.is_popular;  %setting the dependant variable
dropped_columns = {'artist_name', 'track_name', 'Var1', 'track_id', 'popularity', 'genre', 'key', 'mode', 'duration_ms', 'time_signature', 'year'};
%we drop artist name as it can not be read by the model. While the artist
%name may likely have an impact on a songs popularity think Drake vs Lenny
%Breau but without finding a way to quanitfy this it is difficult to
%include. 1 paper suggested to use the average of the song popularity and
%assign it as the artist popularity however I feel doing that will cause
%some bleed as that is what we are trying to predict - https://rpubs.com/annabauer/940476
%we drop track name as there is not much value to be gained from that apart
%from maybe a clickbait name that is so out there it makes you want to play
%the song but again this is difficult to quantify
%we drop the index Var1 but keep the track id as the dataset does not
%contain any duplicates
%genre suffers the same fate as artist name but with the additionaly caveat
%that this is actually a categorical variable but because there are so many
%genres it would introduce too many dummy variables and likely curse us
%with dimensionality
all_columns = logistic_spotify.Properties.VariableNames; %getting a list of all the columns from which I will remove the columns I do not want
kept_columns = all_columns(~ismember(all_columns, dropped_columns)); %removing the columns I do not want
features = logistic_spotify(:, kept_columns); %setting the independant variables
holdout_percent = 0.2;  %80-20 split for training and testing
cv = cvpartition(labels, 'Holdout', holdout_percent, 'Stratify', true); % Create a stratified cvpartition based on 'is popular' labels

% Extract indices for the train and test sets so that we know which rows
% are going in which table and pull the appropriate data
train_index = training(cv);
test_index = test(cv);

% Create train and test sets based on the partition indices above and the
% features created above
training_data_lr = features(train_index, :);
test_data_lr = features(test_index, :); %this data has never been seen and will be used on the final model

%at this point I have still kept the is_popular column ie the target
%because this first split is just to split the data into training and test
%sets. I will then take the training sets and transform them and then run a
%cross validation on them. At that point I will remove the dependant
%variable so that I can compare the precited labels to the actual lables in
%the training set. 
% I am moving is popular to the front so scaling is
%easier to do in the following lines
final_training_labels = training_data_lr(:, 10); %we call it final training label as that is the dependant variable that we are moving around in the table
training_data_lr(:, 10) = [];
training_data_lr = [final_training_labels training_data_lr];

%% More preprocessing for logistic regression classifier

%we are going to scale.
%note that loudness is in decibles which is already log scaled. may make
%sense to inverse it and then scale it because the difference between -60
%and -50 would not be the same as -30 to -20 because of the log
%but since we are robust scaling and negative values are expected this
%should not matter
%since most variables are between 0 and 1 and the others more or less have
%strict bound although there are major outliers in duration minutes we will
%use robust scaling
%duration may not have the strictest bounds given songs from different

%Documentation
%also to learn a bit about decibels https://www.youtube.com/watch?v=XU782Xb9J04
%while the initial 20 is saved for processing from results in this test split I was not sure
%if I should follow the same method for the second split. There can be some
%possible data bleed here. I could not find references to support this
%https://uk.mathworks.com/help/matlab/ref/table.varfun.html
%https://uk.mathworks.com/help/matlab/ref/repmat.html
%https://arxiv.org/pdf/2108.02497.pdf - page 5 - train test data leak

% Calculate median and IQR for each column except the dummies. There is a
% normalize function I could have used
% https://uk.mathworks.com/help/matlab/ref/double.normalize.html with
% medianiqr argument but I wanted to try and code it up myself and seems to
% work okay

training_data_numericals = training_data_lr(:, 2:11);
training_medians = median(training_data_numericals);
%i tired to use percentile on the table but couldnt
%https://uk.mathworks.com/matlabcentral/answers/372363-error-using-sum-invalid-data-type-first-argument-must-be-numeric-or-logical-but-the-data-inside
%eventually found varfun which worked better than looping
training_lower_quartiles = varfun(@(x) prctile(x, 25), training_data_numericals);
training_upper_quartiles = varfun(@(x) prctile(x, 75), training_data_numericals);
training_iqr_values = training_upper_quartiles - training_lower_quartiles;
repeated_medians = repmat(training_medians, size(training_data_numericals, 1), 1); %i had to repeat the rows because robust scaling needed the same size of the table
%originally the medians and iqs were 1x10 tables but the training data is
%like 900k rows
repeated_iqr_values = repmat(training_iqr_values, size(training_data_numericals, 1), 1);

truncated_names = extractAfter(repeated_iqr_values.Properties.VariableNames, 4); % Remove first 4 characters because for some reason fun_ was added to the names
repeated_iqr_values = renamevars(repeated_iqr_values, repeated_iqr_values.Properties.VariableNames, truncated_names);

%saving the training values to scale the test set later to avoid
%bleed/leakage
writetable(training_iqr_values, 'training_iqr_values.csv');
writetable(training_medians, 'training_medians.csv');

% Robust scaling
robust_scaled_training_data_numericals = (training_data_numericals - repeated_medians) ./ repeated_iqr_values;
training_data_dummies = training_data_lr(:, 12:end);
final_training_data = [robust_scaled_training_data_numericals training_data_dummies];

%% Base model logistic regression classifier being trained
%all variables here should be marked 1
%now we are going to fit our training data onto a model
%we will create a classifier that is a logistic regression


% Convert the table to an array since the clothes need to be worn not the
% wardrobe - https://uk.mathworks.com/matlabcentral/answers/372363-error-using-sum-invalid-data-type-first-argument-must-be-numeric-or-logical-but-the-data-inside
final_training_data_array = table2array(final_training_data);
final_training_labels_array = table2array(final_training_labels);

%final_training_data_vector = final_training_data_array(:);
final_training_labels_vector = final_training_labels_array(:);

rng(200); % Setting the random seed for reproducibility

cv1 = cvpartition(final_training_labels_vector, 'KFold', 10, 'Stratify', true);

% Initialize variables to store metrics for each fold
auc_1 = zeros(cv1.NumTestSets, 1);
precision_1 = zeros(cv1.NumTestSets, 1);
recall_1 = zeros(cv1.NumTestSets, 1);
accuracy_1 = zeros(cv1.NumTestSets, 1);
F1_1 = zeros(cv1.NumTestSets, 1);
fit_time_1 = zeros(cv1.NumTestSets, 1);

for fold = 1:cv1.NumTestSets
    train_index_1 = training(cv1, fold);
    test_index_1 = test(cv1, fold);

    training_data_1 = final_training_data_array(train_index_1, :);
    training_labels_1 = final_training_labels_vector(train_index_1);
    testing_data_1 = final_training_data_array(test_index_1, :);
    testing_labels_1 = final_training_labels_vector(test_index_1);

    %to measure time of code section we use tic toc
    %https://uk.mathworks.com/help/matlab/matlab_prog/measure-performance-of-your-program.html
    tic;
    base_logistic_classifier = fitclinear(training_data_1, training_labels_1, 'Learner', 'logistic');
    fit_time_1(fold) = toc;

    predicted_probabilities_1 = predict(base_logistic_classifier, testing_data_1);

    predicted_classes_1 = predicted_probabilities_1 > 0.5; %this is an arbitrary cut, could be optimised for

    %setting up the confusion matrix labels 
    %https://core.ac.uk/download/pdf/216913541.pdf - comparing the
    %following metrics for logistic and random forests
    TP_1 = sum(predicted_classes_1 & testing_labels_1);
    FN_1 = sum(~predicted_classes_1 & testing_labels_1);
    FP_1 = sum(predicted_classes_1 & ~testing_labels_1);
    TN_1 = sum(~predicted_classes_1 & ~testing_labels_1);

    %metric calculations
    precision_1(fold) = TP_1 / (TP_1 + FP_1);
    recall_1(fold) = TP_1 / (TP_1 + FN_1);
    accuracy_1(fold) = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1);
    F1_1(fold) = 2 * (precision_1(fold) * recall_1(fold)) / (precision_1(fold) + recall_1(fold)); %got an error here about element wise multiplication 
    % and then realized this should be calculating for each fold not
    % overall

    %https://uk.mathworks.com/help/stats/perfcurve.html
    [~, ~, ~, auc_1(fold)] = perfcurve(testing_labels_1, predicted_probabilities_1, true);
end

auc_1 = mean(auc_1);
precision_1 = mean(precision_1);
recall_1 = mean(recall_1);
accuracy_1 = mean(accuracy_1);
F1_1 = mean(F1_1);
fit_time_1 = mean(fit_time_1);

base_logistic_classifier_mean_metrics = table();

base_logistic_classifier_mean_metrics.Precision = precision_1;
base_logistic_classifier_mean_metrics.Accuracy = accuracy_1;
base_logistic_classifier_mean_metrics.Recall = recall_1;
base_logistic_classifier_mean_metrics.F1 = F1_1;
base_logistic_classifier_mean_metrics.AUC = auc_1;
base_logistic_classifier_mean_metrics.FitTime = fit_time_1;

writetable(base_logistic_classifier_mean_metrics, 'base_logistic_classifier_mean_metrics.csv');

%% Base logistic classifier being saved

final_base_logistic_classifier = fitclinear(final_training_data_array, final_training_labels_vector, 'Learner', 'logistic');

save('final_base_logistic_classifier.mat', 'final_base_logistic_classifier');

%% Tuned model of the logistic regression classifier being trained
%all variables here will be marked as 2
% the first version of the logistic regression was just a simple model with
% stratified cross validation. Now we will make a more complex model that
% includes iterations, lasso technique and stratified sampling. we
% will then set a range of values for the 2 hyper parameters and implement
% a grid search using loops to iterate through each combination and find
% the best values of the hyperparameters. random search may have been
% better due to not knowing good ranges or computation resources
%we are optimising for precision as I would rather correctly predict which
%song would be popular and reward those songs. 
%we could also find the best precision threshold but that has been left out
%for now
%i had tried to run weights but for some reason was unable to get it to
%work with lassoglm. I had quite a few problems with the documentation of
%matlab
%chose to do lasso and thus optimise for lambda over ridge because I wanted
%to entirely remove the effect of some coefficients ie do some feature
%selection where as ridge only minimises their effect. A different kind f
%regularization might have resulted in different outcomes
%I paired this with the second hyperparameter the number of iterations to
%see how many tries the model would need to converge to the optimal
%coefficients that minimised the log loss
%other hyperparameters were not considered to save on compute time

%Documentation
%https://arxiv.org/pdf/1912.06059.pdf - differences between grid search and
%random search
%https://uk.mathworks.com/help/stats/lassoglm.html - shows available
%hyperparameters 
%https://www.sciencedirect.com/science/article/pii/S1877705817341474 -
%difference between ridge and lasso

rng(200); % For reproducibility
cv_folds_2 = 10; % Define the number of folds for stratified cross-validation
cv2 = cvpartition(final_training_labels_vector, 'KFold', cv_folds_2, 'Stratify', true);
lambdas = logspace(-4, 4, 10); % Define lambda values for tuning
iterations = [25, 50, 100, 150]; % Define iterations values for tuning

% Initialize variables to store AUC and precision for all combinations of iterations and lambdas
auc_2 = zeros(length(lambdas), length(iterations));
precision_2 = zeros(length(lambdas), length(iterations));
recall_2 = zeros(length(lambdas), length(iterations));
accuracy_2 = zeros(length(lambdas), length(iterations));
F1_2 = zeros(length(lambdas), length(iterations));
fit_time_2 = zeros(length(lambdas), length(iterations));

% Initialize table
tuned_logistic_classifier_mean_metrics = table('Size',[numel(lambdas)*numel(iterations), 8],...
    'VariableTypes',{'double','double','double','double','double','double','double','double'},...
    'VariableNames',{'Lambda','Iterations','Precision','Recall','Accuracy','F1', ...
    'AUC', 'FitTime'});

% Counter row indexing
row_1 = 1;

% grid search for hyperparameter tuning
for lambda_index = 1:length(lambdas)
    lambda = lambdas(lambda_index);
    for iteration_index = 1:length(iterations)
        iteration = iterations(iteration_index);
        
        % Perform cross-validated logistic regression for each hyperparameter combination
        auc_values_2 = zeros(cv_folds_2, 1);
        precision_values_2 = zeros(cv_folds_2, 1);
        recall_values_2 = zeros(cv_folds_2, 1);
        accuracy_values_2 = zeros(cv_folds_2, 1);
        F1_values_2 = zeros(cv_folds_2, 1);
        fit_time_values_2 = zeros(cv_folds_2, 1);


        for fold = 1:cv_folds_2
            train_index_2 = training(cv2, fold);
            test_index_2 = test(cv2, fold);
            
            % Get training and testing data for this fold
            training_data_2 = final_training_data_array(train_index_2, :);
            training_labels_2 = final_training_labels_vector(train_index_2);
            testing_data_2 = final_training_data_array(test_index_2, :);
            testing_labels_2 = final_training_labels_vector(test_index_2);
            
            % Train the model using current hyperparameters with Lasso regularization
            % I could have used fitclinear but I was getting errors with
            % the range of values and perfcurve for some reasons and this
            % combination worked so went with this
            tic;
            [B, FitInfo] = lassoglm(training_data_2, training_labels_2, 'binomial', ...
                'Lambda', lambda, ...
                'MaxIter', iteration ...
            );
            fit_time_2 = toc;
            fit_time_values_2(fold) = fit_time_2;

            % Get the coefficients to be used in the prediction
            constant_lambda_index = 1; %i was trying to index and then realized the loop is picking 1 each time so the index would always be 1
            B0 = FitInfo.Intercept(constant_lambda_index);
            coefs = [B0; B(:, constant_lambda_index)];

            % Predict probabilities for the test fold
            predicted_probabilities_2 = glmval(coefs, testing_data_2, 'logit');
            
            % Calculate AUC using perfcurve for this fold
            [~, ~, ~, auc_2] = perfcurve(testing_labels_2, predicted_probabilities_2, true);
            auc_values_2(fold) = auc_2;

            predicted_classes_2 = predicted_probabilities_2 > 0.5;

            TP_2 = sum(predicted_classes_2 & testing_labels_2);
            FN_2 = sum(~predicted_classes_2 & testing_labels_2);
            FP_2 = sum(predicted_classes_2 & ~testing_labels_2);
            TN_2 = sum(~predicted_classes_2 & ~testing_labels_2);

            % Calculate metrics
            precision_2 = TP_2 / (TP_2 + FP_2);
            precision_values_2(fold) = precision_2;

            recall_2 = TP_2 / (TP_2 + FN_2);
            recall_values_2(fold) = recall_2;

            accuracy_2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2);
            accuracy_values_2(fold) = accuracy_2;

            F1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2);
            F1_values_2(fold) = F1_2;
        end
        
        % Calculate mean metrics across folds for this hyperparameter combination
        mean_auc_2 = mean(auc_values_2);
        mean_precision_2 = mean(precision_values_2);
        mean_recall_2 = mean(recall_values_2);
        mean_accuracy_2 = mean(accuracy_values_2);
        mean_F1_2 = mean(F1_values_2);
        mean_fit_time_2 = mean(fit_time_values_2);

        % Store mean AUC and precision values for this combination of lambda and iteration
        auc_2 = mean_auc_2;
        precision_2 = mean_precision_2;
        recall_2 = mean_recall_2;
        accuracy_2 = mean_accuracy_2;
        F1_2 = mean_F1_2;
        fit_time_2 = mean_fit_time_2;

       row_values_1 = {lambdas(lambda_index), iterations(iteration_index), precision_2, recall_2, accuracy_2, ...
                 F1_2, auc_2, fit_time_2}; %chatgpt to debug because I could not figure out how to assign the values directly. 
        % Needed to find a conversion
        tuned_logistic_classifier_mean_metrics(row_1,:) = row_values_1;
        row_1 = row_1 + 1;
    end
end

writetable(tuned_logistic_classifier_mean_metrics, 'tuned_logistic_classifier_mean_metrics.csv');

max_precision = max(tuned_logistic_classifier_mean_metrics.Precision);
tuned_logistic_classifier_best_mean_metrics = tuned_logistic_classifier_mean_metrics ...
    (tuned_logistic_classifier_mean_metrics.Precision == max_precision, :);
%the above retured more than 1 row incase of ties. Just for ease I picked
%the first row from the table. Alternatives could be writing conditions
%going down the metrics and then at the final tie picking the first row

tuned_logistic_classifier_best_mean_metrics = tuned_logistic_classifier_best_mean_metrics(1,:);

best_lambda = tuned_logistic_classifier_best_mean_metrics.Lambda;
best_iteration = tuned_logistic_classifier_best_mean_metrics.Iterations;

writetable(tuned_logistic_classifier_best_mean_metrics, 'tuned_logistic_classifier_best_mean_metrics.csv');

%% final tuned logistic model being saved
% Train the model using best parameters
[B, FitInfo] = lassoglm(final_training_data_array, final_training_labels_vector, 'binomial', ...
    'Lambda', best_lambda, ...
    'MaxIter', best_iteration ...
    );

% Final coefficients to be used in the prediction
constant_lambda_index = 1;
final_B0 = FitInfo.Intercept(constant_lambda_index);
final_coefs = [B0; B(:, constant_lambda_index)];

%final probablities
final_tuned_logistic_classifier = fitglm(final_training_data_array, final_training_labels_vector, ...
    'Distribution', 'binomial', ...
    'Link', 'logit', ...
    'B0', final_coefs); %coefficients from lassoglm

% Save model
save('final_tuned_logistic_classifier.mat', 'final_tuned_logistic_classifier');


%initially i ran the model with 3 iterations, 50 100 and 150 and the best
%iteration was apparently 50. This made me think that maybe the model is
%converging a lot faster than 50 and i included one more interation
%parameter which was 25 and the best iteration then became 25. The mean auc
%scores also flattened out and were no worse than flipping a coin after the
%first few combinations which means that in the given defined spaces for
%lambda and iterations no further improvement can be made. This is shown in
%the mean precision values as well which surprisngly after the first few
%combinations was no longer predicting any positive classes (nans are
%populated indicating a division by 0). This can further mean that the
%variables them selves are not predictive enough for popularity and some
%other feature engineering is required

%% Random Forest - similar steps as above expect less pre processing
%all variables here will be marked as 3
% i do not need to normalize
% i do not need to create dummies
% i do need to split into train and test
% for the cateogircal variables that are numbers we will change them to categorical
% could not find papers for the above but did this just to be safe to avoid the model thinking and order exists

%Documentation
%https://link.springer.com/article/10.1023/A:1010933404324#preview - random
%forest, page 8, bagging is unbiased but likely needs a large number of
%trees to do cross validation for free
%https://www.youtube.com/watch?v=-uPiqnjgVeA same as above but see around
%the 10 minute mark
%https://journals.sagepub.com/doi/10.1177/1536867X20909688 - more random
%forests
%https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050737/ - LR vs RF, I expect
%RF to be better
%https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6368971/#:~:text=One%20reason%20for%20the%20widespread,handle%20ordinal%20and%20nominal%20predictors.
%- no need for dummies natively
%https://www.frontiersin.org/articles/10.3389/fbioe.2020.00855/full -
%normalization not the most important as it is not a distance based
%algorithm

randomf_spotify = spotify; %here i am making a copy of the original table and this is what we will work with
randomf_spotify.duration_minutes = round(randomf_spotify.duration_ms / 60000,1); %convert milliseconds to minutes and round to 1 decimal place
randomf_spotify = convertvars(randomf_spotify,{'key','mode','time_signature'},'categorical'); %converting to categorical found nicer code on matlab

rng(2023);  %setting default seed as before for reproducability
labels_3 = randomf_spotify.is_popular;  %setting the dependant variable
dropped_columns_3 = {'artist_name', 'track_name', 'Var1', 'track_id', 'popularity', 'genre', 'duration_ms', 'year'}; 
all_columns_3 = randomf_spotify.Properties.VariableNames; %getting a list of all the columns from which I will remove the columns I do not want
kept_columns_3 = all_columns_3(~ismember(all_columns_3, dropped_columns_3)); %removing the columns I do not want
features_3 = randomf_spotify(:, kept_columns_3); %setting the independant variables
holdout_percent_3 = 0.2;  %80-20 split for training and testing
cv3 = cvpartition(labels_3, 'Holdout', holdout_percent_3, 'Stratify', true); % Create a stratified cvpartition based on 'is popular' labels

train_index_3 = training(cv3);
test_index_3 = test(cv3);

% Create train and test sets based on the partition indices above and the
% features created above
training_data_rf = features_3(train_index_3, :);
test_data_rf = features_3(test_index_3, :); %this data is never seen by the algorithm

final_training_labels_rf = training_data_rf(:, 13); %we call it final training label as that is the dependant variable that we are moving around in the table
training_data_rf(:, 13) = [];
final_training_data_rf = training_data_rf;
final_training_labels_rf_array = table2array(final_training_labels_rf);

%% Base Line Random Forest

% Splitting the data into training and testing sets
rng(200); % For reproducibility
cv4 = cvpartition(final_training_labels_rf_array, 'KFold', 10, 'Stratify', true);
treesnum = 100; %just an initial value

% Initialize variables to store metrics for each fold
auc_3 = zeros(cv4.NumTestSets, 1);
precision_3 = zeros(cv4.NumTestSets, 1);
recall_3 = zeros(cv4.NumTestSets, 1);
accuracy_3 = zeros(cv4.NumTestSets, 1);
F1_3 = zeros(cv4.NumTestSets, 1);
fit_time_3 = zeros(cv4.NumTestSets, 1);

for fold = 1:cv4.NumTestSets
    train_index_4 = training(cv4, fold);
    test_index_4 = test(cv4, fold);
    
    training_data_3 = final_training_data_rf(train_index_4, :);
    training_labels_3 = final_training_labels_rf_array(train_index_4);
    testing_data_3 = final_training_data_rf(test_index_4, :);
    testing_labels_3 = final_training_labels_rf_array(test_index_4);
    
    % Creating a Random Forest classifier using the training data
    tic;
    base_random_forest_classifier = TreeBagger(treesnum, training_data_3, training_labels_3, 'Method', 'classification');
    fit_time_3(fold) = toc;

    % Making predictions on the testing data
    predicted_classes_3 = predict(base_random_forest_classifier, testing_data_3);

    %https://uk.mathworks.com/help/matlab/ref/cellfun.html
    %https://uk.mathworks.com/help/matlab/ref/strcmp.html
    predicted_numeric = cellfun(@(x) double(strcmp(x, '1')), predicted_classes_3);
    %had weird problems here because the variables were outputting 0s and
    %1s but the error I was getting was that positive class does not exist
    %even though I could open the variables and see that the 0 and 1s were
    %actually there. I then figured it might be a data type problem similar
    %to what happens in excel or python where it looks like a number but is
    %maybe a string. so I used the cellfun and strcmp to see what would
    %have happen. But then this code did not work with the confusion matrix
    %calcualtions so then used chatgpt to convert it to a double which
    %seemed to do the trick

    TP_3 = sum(predicted_numeric & testing_labels_3);
    FN_3 = sum(~predicted_numeric & testing_labels_3);
    FP_3 = sum(predicted_numeric & ~testing_labels_3);
    TN_3 = sum(~predicted_numeric & ~testing_labels_3);
    
    precision_3(fold) = TP_3 / (TP_3 + FP_3);
    recall_3(fold) = TP_3 / (TP_3 + FN_3);
    accuracy_3(fold) = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3);
    F1_3(fold) = 2 * (precision_3(fold) * recall_3(fold)) / (precision_3(fold) + recall_3(fold));
    [~, ~, ~, auc_3(fold)] = perfcurve(testing_labels_3, predicted_numeric, 1); %https://uk.mathworks.com/matlabcentral/answers/163600-roc-curve-with-matlab-using-svmtrain
end

auc_3 = mean(auc_3);
precision_3 = mean(precision_3);
recall_3 = mean(recall_3);
accuracy_3 = mean(accuracy_3);
F1_3 = mean(F1_3);
fit_time_3 = mean(fit_time_3);

base_random_forest_classifier_mean_metrics = table();

base_random_forest_classifier_mean_metrics.Precision = precision_3;
base_random_forest_classifier_mean_metrics.Accuracy = accuracy_3;
base_random_forest_classifier_mean_metrics.Recall = recall_3;
base_random_forest_classifier_mean_metrics.F1 = F1_3;
base_random_forest_classifier_mean_metrics.AUC = auc_3;
base_random_forest_classifier_mean_metrics.FitTime = fit_time_3;

writetable(base_random_forest_classifier_mean_metrics, 'base_random_forest_classifier_mean_metrics.csv');
% base rf is already out predicting across all scores vs lr


%% final base random forest model being saved

final_base_random_forest_classifier = TreeBagger(treesnum, final_training_data_rf, final_training_labels_rf_array, 'Method', 'classification');
save('final_base_random_forest_classifier.mat', 'final_base_random_forest_classifier', '-v7.3'); %using v7.3 in case the model is too large to save in normal mat file

%% Tuned random forest
% hyperparameter optimized forest
% all variables here will be marked as 4

rng(200);
cv_folds_4 = 10;
cv5 = cvpartition(final_training_labels_rf_array, 'KFold', cv_folds_4, 'Stratify', true);

treesnum1 = [50, 100, 150, 200]; % Different numbers of trees
minleafs = [5, 10, 15, 20]; % Different minimum leaf sizes

% Initialize variables to store results
%best_oob_error = 0;

auc_4 = zeros(length(treesnum1), length(minleafs));
precision_4 = zeros(length(treesnum1), length(minleafs));
recall_4 = zeros(length(treesnum1), length(minleafs));
accuracy_4 = zeros(length(treesnum1), length(minleafs));
F1_4 = zeros(length(treesnum1), length(minleafs));
fit_time_4 = zeros(length(treesnum1), length(minleafs));
oob_error_4 = zeros(length(treesnum1), length(minleafs));

tuned_random_forest_classifier_mean_metrics = table('Size',[numel(treesnum1)*numel(minleafs), 9],...
    'VariableTypes',{'double','double','double','double','double','double','double','double', 'double'},...
    'VariableNames',{'Number_of_Trees','Minimum_Leaf_Size','Precision','Recall','Accuracy','F1', ...
    'AUC','OOBError', 'FitTime'});

% Counter row indexing
row_2 = 1;


% grid search
for i = 1:length(treesnum1)
    for j = 1:length(minleafs)
        % TreeBagger

        auc_values_4 = zeros(cv_folds_4, 1);
        precision_values_4 = zeros(cv_folds_4, 1);
        recall_values_4 = zeros(cv_folds_4, 1);
        accuracy_values_4 = zeros(cv_folds_4, 1);
        F1_values_4 = zeros(cv_folds_4, 1);
        fit_time_values_4 = zeros(cv_folds_4, 1);
        oob_error_values_4 = zeros(cv_folds_4, 1);

        for fold = 1:cv5.NumTestSets
            train_index_5 = training(cv5, fold);
            test_index_5 = test(cv5, fold);

            training_data_4 = final_training_data_rf(train_index_5, :);
            training_labels_4 = final_training_labels_rf_array(train_index_5);
            testing_data_4 = final_training_data_rf(test_index_5, :);
            testing_labels_4 = final_training_labels_rf_array(test_index_5);

            tic;
            tuned_random_forest_classifier = TreeBagger(treesnum1(i), training_data_4, training_labels_4, 'Method', 'classification', ...
                'MinLeafSize', minleafs(j), 'OOBPrediction', 'On');
            fit_time_4 = toc;
            fit_time_values_4(fold) = fit_time_4;
    
            % OOB predictions
            [oobPredictions, oobScores] = oobPredict(tuned_random_forest_classifier);
            oob_predicted_numeric = cellfun(@(x) double(strcmp(x, '1')), oobPredictions);
    
            % OOB error
            %err = oobError(rfmodel1); %kept getting this error Unable to use a value of type TreeBagger as an index. 
            % even though the documentation uses it like this https://uk.mathworks.com/help/stats/treebagger.ooberror.html
            oob_error_4 = sum(oob_predicted_numeric ~= training_labels_4) / numel(training_labels_4);%chatgpt eventually
            oob_error_values_4(fold) = oob_error_4;

            % and https://www.analyticsvidhya.com/blog/2020/12/out-of-bag-oob-score-in-the-random-forest-algorithm/
    
            predicted_classes_4 = predict(tuned_random_forest_classifier, testing_data_4);
            
            predicted_numeric_1 = cellfun(@(x) double(strcmp(x, '1')), predicted_classes_4);

            TP_4 = sum(predicted_numeric_1 & testing_labels_4);
            FN_4 = sum(~predicted_numeric_1 & testing_labels_4);
            FP_4 = sum(predicted_numeric_1 & ~testing_labels_4);
            TN_4 = sum(~predicted_numeric_1 & ~testing_labels_4);
            
            precision_4 = TP_4 / (TP_4 + FP_4);
            precision_values_4(fold) = precision_4;

            recall_4 = TP_4 / (TP_4 + FN_4);
            recall_values_4(fold) = recall_4;

            accuracy_4 = (TP_4 + TN_4) / (TP_4 + TN_4 + FP_4 + FN_4);
            accuracy_values_4(fold) = accuracy_4;

            F1_4 = 2 * (precision_4 * recall_4) / (precision_4 + recall_4);
            F1_values_4(fold) = F1_4;

            [~, ~, ~, auc_4] = perfcurve(testing_labels_4, predicted_numeric_1, 1);
            auc_values_4(fold) = auc_4;

        end

        mean_auc_4 = mean(auc_values_4);
        mean_precision_4 = mean(precision_values_4);
        mean_recall_4 = mean(recall_values_4);
        mean_accuracy_4 = mean(accuracy_values_4);
        mean_F1_4 = mean(F1_values_4);
        mean_fit_time_4 = mean(fit_time_values_4);
        mean_oob_error_4 = mean(oob_error_values_4);

        auc_4 = mean_auc_4;
        precision_4 = mean_precision_4;
        recall_4 = mean_recall_4;
        accuracy_4 = mean_accuracy_4;
        F1_4 = mean_F1_4;
        fit_time_4 = mean_fit_time_4;
        oob_error_4 = mean_oob_error_4;

        row_values_2 = {treesnum1(i), minleafs(j), precision_4, recall_4, accuracy_4, ...
                 F1_4, auc_4, oob_error_4, fit_time_4}; %chatgpt to debug because I could not figure out how to assign the values directly. 
        % Needed to find a conversion
        tuned_random_forest_classifier_mean_metrics(row_2,:) = row_values_2;
        row_2 = row_2 + 1;
    end
end

writetable(tuned_random_forest_classifier_mean_metrics, 'tuned_random_forest_classifier_mean_metrics.csv');

min_OOBError = min(tuned_random_forest_classifier_mean_metrics.OOBError);
tuned_random_forest_classifier_best_mean_metrics = tuned_random_forest_classifier_mean_metrics ...
    (tuned_random_forest_classifier_mean_metrics.OOBError == min_OOBError, :);

tuned_random_forest_classifier_best_mean_metrics = tuned_random_forest_classifier_best_mean_metrics(1,:);
%picking the above row same as previous reason

besttreesnum = tuned_random_forest_classifier_best_mean_metrics.Number_of_Trees;
bestminleafs = tuned_random_forest_classifier_best_mean_metrics.Minimum_Leaf_Size;


writetable(tuned_random_forest_classifier_best_mean_metrics, 'tuned_random_forest_classifier_best_mean_metrics.csv');

% Initialize subplots taken from gpt since the code from the documentation
% was not allowing me to reference the model like it does in the
% documentation https://uk.mathworks.com/help/stats/treebagger.html

oob_error_plot = figure('Position', [100, 100, 1000, 800]);

for j = 1:length(minleafs)
    subplot(2, 2, j);
    
    % Extract OOB errors and number of trees for the current minleaf value
    subset_results = tuned_random_forest_classifier_mean_metrics(tuned_random_forest_classifier_mean_metrics{:, 'Minimum_Leaf_Size'} == minleafs(j), :);
    
    % Plot OOB errors
    plot(subset_results{:, 'Number_of_Trees'}, subset_results{:, 'OOBError'}, '-o');
    
    title(['Minimum Leaf Size = ' num2str(minleafs(j))]);
    xlabel('Number of Trees');
    ylabel('Out-of-Bag Error');
    grid on;
end
sgtitle('Average OOB Errors vs Number of Trees for Different Minimum Leaf Sizes');
saveas(oob_error_plot, 'Average OOB Errors vs Number of Trees for Different Minimum Leaf Sizes.png');

%% final random forest model

final_tuned_random_forest_classifier = TreeBagger(besttreesnum, final_training_data_rf, final_training_labels_rf_array, 'Method', 'classification', ...
            'MinLeafSize', bestminleafs, 'OOBPrediction', 'On');

save('final_tuned_random_forest_classifier.mat', 'final_tuned_random_forest_classifier', '-v7.3');

%% saving the test sets

writetable(test_data_lr, 'test_data_lr.csv');
writetable(test_data_rf, 'test_data_rf.csv');

%% -----------CODE IS GOOD TILL HERE----------- %%
