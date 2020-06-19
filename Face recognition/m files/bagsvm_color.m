clc;
clear all;

tic;
path = fullfile('/Users/aishwarya/Documents/City MSc Data Science/Term 2/Computer Vision/Models/color_models/Models_27th');
imgFolder = fullfile(path, 'bag_svm');

faceDatabase = imageSet(imgFolder, 'recursive');

% Split Database into Training & Test Sets
rng(6);
[training,test] = partition(faceDatabase,[0.8 0.2], 'random');

% Returns a bag of features object for the training set
bag = bagOfFeatures(training);

% Extract Bag-of-Words Features for Training Set

training_sets = numel(training);    % number of categories (69)
training_set_size = sum([training.Count]);       % total number of training images
trainingFeatures = encode(bag,training);      % feature matrix of visual words for the training set
featureCount = 1;


% loop through each image, extract and add HOG Features to trainingFeatures
% matrix, and create vector holding label data per image
for i=1:training_sets
    label = training(i).Description;
    for j = 1:training(i).Count
        trainingLabel{featureCount} = label;
        featureCount = featureCount + 1;
    end
end

%Train SVM using extracted HOG features and class labels 1-81
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

% Extract Bag-of-Words Features for Test Set

testSets = numel(test);         % number of categories (69)
testSetSize = sum([test.Count]);        % total number of test images

testFeatureCount = 1;

% creates feature vector that represents a histogram of visual word 
% occurrences from the test set
testFeatures = encode(bag,test);

% loop through each image in the test set and record the labels
for i=1:testSets
    actualLabel = test(i).Description;
    for j=1:test(i).Count
        actualTestLabels{testFeatureCount, :} = actualLabel;
        testFeatureCount = testFeatureCount + 1;
    end
end

%Predict matching labels for all images in the test set

testLabels = predict(faceClassifier, testFeatures);

%Calculate accuracy of test imageset

correctMatches = 0;

% Check whether predicted label matches actual label for each image, and
% count how many are equivalent
for i=1:testSetSize
    if strcmp(testLabels{i}, actualTestLabels(i,:))
       correctMatches = correctMatches + 1;
    end
end

% calculate accuracy
accuracy = correctMatches/testSetSize

% Train a classifier with the Training Set
categoryClassifier = trainImageCategoryClassifier(training, bag); 

% Evaluate the classifier using the Test Set
confMatrix = evaluate(categoryClassifier, test); 

% Compute average accuracy
accuracy1 = mean(diag(confMatrix))
toc;

save bag_svm_model.mat;